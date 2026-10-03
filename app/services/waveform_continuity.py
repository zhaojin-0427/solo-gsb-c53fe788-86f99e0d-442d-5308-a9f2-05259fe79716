"""Station waveform acquisition continuity check.

A read-only cross-ingestion view answering "for one station, which parts of a
requested UTC half-open window did the platform actually acquire, and where did
acquisition ranges overlap?":

* The client submits a station code and a half-open UTC window
  ``[start_time, end_time)``.  The service reads that station's
  :class:`~app.models.Waveform` metadata across **all** ingestion batches --
  runs, events, picks and on-disk directories play no role.
* Each waveform segment is converted from its start sample and
  ``npts / sampling_rate`` into a half-open coverage interval
  ``[start, start + npts/sr)`` and clipped to the request window.
* Waveforms of the same station whose start, sampling rate, sample count and
  data hash are all identical are the same acquired segment and count once;
  every ingestion id carrying that segment is retained so re-ingestions stay
  auditable.
* Endpoints are swept left to right and each open interior is classified by
  the number of **unique** segments covering it: ``0`` deep means **gap**,
  ``>= 1`` deep means **covered** (the coverage union), and the ``>= 2`` deep
  subintervals are additionally listed as **overlaps** (overlap means
  acquisition ranges coincide -- nothing is inferred about sample-content
  conflicts).  Coverage segments are therefore the union of everything
  acquired and *include* their overlap subintervals; the two lists overlap in
  time but never double-count anything in ``totals``.
* Consecutive covered cells are merged even when their depth changes across
  the boundary (e.g. depth 2 abutting depth 1 stays one coverage segment), as
  are consecutive overlap-only and gap cells.  Every ingestion id covering any
  part of a merged segment is retained, and ``coverage_count`` is the highest
  unique-segment depth reached inside the segment.
* Gaps are the uncovered interior subintervals, classified as leading,
  trailing or internal; a window with no records for a known station yields a
  single whole-window gap.  A station that was never ingested returns 404.

Invariant: ``covered_s + gap_s == window_s``; ``overlap_s`` is the portion of
``covered_s`` whose depth is >= 2 (never an addend in the partition).

The check is purely derived and never writes state, so its result is
independent of run/directory state; ingestion and pick-gap reports are
unaffected.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Waveform
from .pipeline import ApiError, from_epoch, iso_z, to_epoch, to_naive_utc

# Half-open interval classes.  A cell's class is the depth of unique-segment
# coverage over its open interior; CLASS_COVERED means "in the coverage union"
# (depth >= 1) and the depth>=2 cells of that union are mirrored into the
# separate ``overlaps`` list.  CLASS_OVERLAP cells thus appear in *both* the
# merged coverage segments and the merged overlap segments.
CLASS_GAP = "gap"          # zero unique segments cover the interior
CLASS_COVERED = "covered"  # >= 1 unique segment covers the interior (union)
CLASS_OVERLAP = "overlap"  # >= 2 unique segments cover the interior

# Segment endpoints closer than this are the same sample instant computed two
# ways (stored microsecond start vs. start + npts/sr); cluster them so abutting
# acquisitions merge instead of showing a spurious nanosecond gap.
EPS_S = 1e-9


def continuity_check(
    db: Session, station: str, start_time: str | None, end_time: str | None
) -> dict:
    """Build the coverage/gap/overlap report for one station and window."""
    start = _parse_window_bound(start_time, "start_time")
    end = _parse_window_bound(end_time, "end_time")
    if start >= end:
        raise ApiError(
            422,
            "invalid_time_window",
            "time window must be a non-empty half-open interval "
            "(start_time < end_time)",
        )

    w0, w1 = to_epoch(start), to_epoch(end)

    rows = (
        db.query(Waveform)
        .filter(Waveform.station == station)
        # Numeric order also fixes the presentation order of equal endpoints.
        .order_by(Waveform.start_time, Waveform.sampling_rate, Waveform.npts,
                  Waveform.data_sha256, Waveform.ingestion_id)
        .all()
    )
    if not rows:
        raise ApiError(
            404,
            "station_not_found",
            f"station {station} has never been ingested (no waveform metadata)",
        )

    # Deduplicate identical acquired segments (same start / sampling rate /
    # npts / data hash) while retaining every ingestion id that carries them.
    groups: dict[tuple, dict] = {}
    for wf in rows:
        seg_start = to_epoch(wf.start_time)
        seg_end = seg_start + int(wf.npts) / float(wf.sampling_rate)
        # Clip to the half-open request window.  The dedup key is the
        # *original* segment identity, not the clipped remainder.
        c0, c1 = max(seg_start, w0), min(seg_end, w1)
        if c1 <= c0:
            continue  # touches the window at most at one endpoint
        key = (wf.start_time, wf.sampling_rate, int(wf.npts), wf.data_sha256)
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "start": c0,
                "end": c1,
                "ingestion_ids": {wf.ingestion_id},
                # Identity coordinates for stable presentation / debugging.
                "sampling_rate": float(wf.sampling_rate),
                "npts": int(wf.npts),
            }
        else:
            # Two identical original segments clip identically (the key fixes
            # the original interval); union the clipped ranges defensively.
            group["start"] = min(group["start"], c0)
            group["end"] = max(group["end"], c1)
            group["ingestion_ids"].add(wf.ingestion_id)

    segments = [
        {"start": g["start"], "end": g["end"],
         "ingestion_ids": sorted(g["ingestion_ids"])}
        for g in groups.values()
    ]
    segments.sort(key=lambda s: (s["start"], s["end"], s["ingestion_ids"][0]))

    raw_intervals = _sweep(w0, w1, segments)
    # Coverage is the depth>=1 union: merge runs of consecutive non-gap cells
    # (depth may rise above and fall back across abutments).  Overlaps are the
    # depth>=2 subintervals of that same union, merged on their own, so the
    # two lists overlap in time but totals never double-count their seconds.
    covered = _merge_coverage(raw_intervals)
    overlaps = _merge_abutting(
        [iv for iv in raw_intervals if iv["kind"] == CLASS_OVERLAP]
    )
    gaps = _merge_abutting(
        [iv for iv in raw_intervals if iv["kind"] == CLASS_GAP]
    )
    _label_gaps(gaps, w0, w1)

    duplicates = sorted(
        (
            {
                "start_time": iso_z(key[0]),
                "sampling_rate": float(key[1]),
                "npts": int(key[2]),
                "data_sha256": key[3],
                "ingestion_ids": sorted(g["ingestion_ids"]),
                "ingestion_count": len(g["ingestion_ids"]),
            }
            for key, g in groups.items()
            if len(g["ingestion_ids"]) > 1
        ),
        key=lambda d: (d["start_time"], d["sampling_rate"], d["npts"],
                       d["data_sha256"]),
    )

    # Partition of the window: covered union + gaps == window_s.  overlap_s is
    # the depth>=2 part already counted in covered_s, not a third addend.
    covered_s = sum(iv["end"] - iv["start"] for iv in covered)
    overlap_s = sum(iv["end"] - iv["start"] for iv in overlaps)
    gap_s = sum(iv["end"] - iv["start"] for iv in gaps)

    return {
        "station": station,
        "window": {
            "start": iso_z(start),
            "end": iso_z(end),
            "interval": "half_open",
        },
        "ingested_waveform_count": len(rows),
        "unique_segment_count": len(groups),
        "duplicate_waveform_count": len(duplicates),
        "coverage": [_segment_payload(iv) for iv in covered],
        "gaps": [_gap_payload(iv) for iv in gaps],
        "overlaps": [_segment_payload(iv) for iv in overlaps],
        "duplicates": duplicates,
        "totals": {
            "covered_s": round(covered_s, 6),
            "gap_s": round(gap_s, 6),
            "overlap_s": round(overlap_s, 6),
            "window_s": round(w1 - w0, 6),
            "covered_segment_count": len(covered),
            "gap_count": len(gaps),
            "overlap_segment_count": len(overlaps),
        },
        # Overlap is a property of acquisition ranges only; the platform never
        # compares sample contents here.
        "overlap_semantics": "acquisition_range_only",
    }


# --------------------------------------------------------------------------- #
# window parsing
# --------------------------------------------------------------------------- #
def _parse_window_bound(value: str | None, field: str) -> datetime:
    if value is None or not str(value).strip():
        raise ApiError(
            422,
            "invalid_time_window",
            f"missing required UTC window bound '{field}'",
        )
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(
            422,
            "invalid_time_window",
            f"'{field}' is not a valid UTC ISO 8601 timestamp: {value!r}",
        ) from exc
    return to_naive_utc(dt)


# --------------------------------------------------------------------------- #
# endpoint sweep over half-open intervals
# --------------------------------------------------------------------------- #
def _sweep(win_start: float, win_end: float, segments: list[dict]) -> list[dict]:
    """Partition ``[win_start, win_end)`` into raw depth cells.

    All segment and window endpoints become partition coordinates; coincident
    coordinates (within :data:`EPS_S`, i.e. "same sample instant" computed two
    ways) are clustered so abutting acquisitions do not produce phantom gaps.
    Membership of each open interior is evaluated against every segment's
    half-open range: a segment counts when ``start <= left`` and
    ``right <= end``, which mirrors the sweep (at the left boundary the
    segment's start already fired and its end has not).  Depth ``0`` is a gap;
    every depth ``>= 1`` cell is tagged covered (the union), and the depth
    ``>= 2`` cells additionally carry the overlap tag so the caller can build
    the coverage union and the overlap subintervals from the same cells.
    Segment counts per station window are small, so the direct predicate
    stays clear.
    """
    coords = sorted(
        {win_start, win_end}
        | {c for seg in segments for c in (seg["start"], seg["end"])}
    )
    clustered: list[float] = []
    for coord in coords:
        if clustered and abs(coord - clustered[-1]) <= EPS_S:
            continue
        clustered.append(coord)

    intervals: list[dict] = []
    for c0, c1 in zip(clustered, clustered[1:]):
        if c1 - c0 < EPS_S:
            continue
        active_ids = {
            idx
            for idx, seg in enumerate(segments)
            if seg["start"] <= c0 + EPS_S and seg["end"] >= c1 - EPS_S
        }
        count = len(active_ids)
        kind = (
            CLASS_OVERLAP if count >= 2
            else CLASS_COVERED if count == 1
            else CLASS_GAP
        )
        intervals.append(
            {
                "kind": kind,
                "start": c0,
                "end": c1,
                "ingestion_ids": _union_ingestion_ids(segments, active_ids),
                "coverage_count": count,
            }
        )
    return intervals


def _union_ingestion_ids(segments: list[dict], active_ids: set[int]) -> list[str]:
    ids: set[str] = set()
    for idx in active_ids:
        ids.update(segments[idx]["ingestion_ids"])
    return sorted(ids)


def _merge_coverage(intervals: list[dict]) -> list[dict]:
    """Merge the depth>=1 cells into coverage-union segments.

    Consecutive non-gap cells belong to one coverage segment even when their
    depth changes across the abutment (depth 1 -> 2 -> 1 is one continuous
    acquired range); only a gap cell breaks the run.  The merged segment keeps
    *every* ingestion covering any part and reports the highest unique depth
    reached inside it as ``coverage_count``.
    """
    merged: list[dict] = []
    for iv in intervals:
        if iv["kind"] == CLASS_GAP:
            continue
        if (
            merged
            and iv["start"] - merged[-1]["end"] <= EPS_S
        ):
            prev = merged[-1]
            prev["end"] = iv["end"]
            ids = set(prev["ingestion_ids"]) | set(iv["ingestion_ids"])
            prev["ingestion_ids"] = sorted(ids)
            prev["coverage_count"] = max(
                prev["coverage_count"], iv["coverage_count"]
            )
        else:
            merged.append(dict(iv))
            # A coverage segment is the union; its tag stays "covered" even
            # when its first cell is a depth>=2 overlap cell.
            merged[-1]["kind"] = CLASS_COVERED
    return merged


def _merge_abutting(intervals: list[dict]) -> list[dict]:
    """Merge consecutive intervals of the same class (abutting ranges)."""
    merged: list[dict] = []
    for iv in intervals:
        if (
            merged
            and merged[-1]["kind"] == iv["kind"]
            and iv["start"] - merged[-1]["end"] <= EPS_S
        ):
            prev = merged[-1]
            prev["end"] = iv["end"]
            # Same coverage class => the union keeps every source ingestion
            # that covers either side; coverage_count takes the deeper cell.
            ids = set(prev["ingestion_ids"]) | set(iv["ingestion_ids"])
            prev["ingestion_ids"] = sorted(ids)
            prev["coverage_count"] = max(prev["coverage_count"], iv["coverage_count"])
        else:
            merged.append(dict(iv))
    return merged


def _label_gaps(gaps: list[dict], win_start: float, win_end: float) -> None:
    for gap in gaps:
        at_start = abs(gap["start"] - win_start) <= EPS_S
        at_end = abs(gap["end"] - win_end) <= EPS_S
        if at_start and at_end:
            gap["gap_kind"] = "whole"
        elif at_start:
            gap["gap_kind"] = "leading"
        elif at_end:
            gap["gap_kind"] = "trailing"
        else:
            gap["gap_kind"] = "internal"


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _segment_payload(iv: dict) -> dict:
    return {
        "start": iso_z(from_epoch(iv["start"])),
        "end": iso_z(from_epoch(iv["end"])),
        "duration_s": round(iv["end"] - iv["start"], 6),
        "coverage_count": iv["coverage_count"],
        "ingestion_ids": iv["ingestion_ids"],
    }


def _gap_payload(iv: dict) -> dict:
    return {
        "start": iso_z(from_epoch(iv["start"])),
        "end": iso_z(from_epoch(iv["end"])),
        "duration_s": round(iv["end"] - iv["start"], 6),
        "gap_kind": iv["gap_kind"],
    }
