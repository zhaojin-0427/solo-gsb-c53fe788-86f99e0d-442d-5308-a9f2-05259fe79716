"""Station P-wave pick gap assessment over a published catalog compilation.

The assessment answers a narrow monitoring question -- "at the stations that
*should* have recorded a published earthquake, did this platform actually
capture a P pick?" -- directly from the frozen catalog evidence:

* Only catalog entries that were **published at submission time** of the
  compilation contribute, and each entry contributes **exactly once**, through
  its master ``(run, event)``.  Draft or withdrawn entries are invisible; the
  entry revisions are frozen with the report.
* The predicted P arrival uses the **master run's** velocity model and
  calibration version: ``origin_time + P_travel_time(layered model, event
  depth + station elevation, epicentral distance) + station P calibration
  offset`` (the calibration offset is 0 for uncalibrated runs).
* A ``(entry, station)`` pair is **observable** only when the run's original
  ingested waveform for that station completely covers
  ``predicted +/- arrival_tolerance_s`` (first to last sample).
* An observable pair is a **hit** when the *same ingestion* carries a manual P
  pick or an STA/LTA candidate inside the window; otherwise it is a **gap**.
  Manual S picks never match, even when they fall inside the window.
* Per station the report gives observable / gap counts and the gap rate;
  stations with fewer than ``min_observable_events`` observable entries are
  flagged as evidence-insufficient and are not scored.

Every per-entry, per-station decision carries its full evidence (prediction,
coverage interval, matching picks or the unobservable reason).  Reports are
immutable and idempotent (canonical request hash), concurrent identical
requests collapse onto one row, and the per-event evidence supports stable
keyset pagination.  Like the catalog, station audit, focal mechanism
inversion and location uncertainty assessment this layer never modifies
ingestions/runs/events/picks or catalog content.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    CalibrationTerm,
    CatalogCompilation,
    CatalogEntry,
    Event,
    Pick,
    PickGapAssessment,
    PickGapEvent,
    PickGapStation,
    Run,
    StationCalibration,
    VelocityModel,
    Waveform,
)
from ..schemas import PickGapCreate
from .association import distance_km
from .pipeline import (
    ApiError,
    canonical_layers,
    from_epoch,
    iso_z,
    to_epoch,
)
from .traveltime import LayeredModel

STATUS_OBSERVABLE = "observable"
STATUS_UNOBSERVABLE = "unobservable"
OUTCOME_HIT = "hit"
OUTCOME_GAP = "gap"

# Structured unobservable reason: the station did take part in the ingestion
# (otherwise it is not part of the assessed station universe at all) but its
# recorded window does not fully cover predicted arrival +/- tolerance.
REASON_WAVEFORM_NOT_COVERED = "waveform_not_covered"

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# assessment creation (select published -> predict/observe -> persist; idempotent)
# --------------------------------------------------------------------------- #
def create_assessment(db: Session, req: PickGapCreate) -> tuple[dict, bool]:
    compilation = db.get(CatalogCompilation, req.compilation_id)
    if compilation is None:
        raise ApiError(
            404,
            "catalog_compilation_not_found",
            f"catalog compilation {req.compilation_id} not found",
        )

    canon = canonical_request(req)
    request_hash = hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()
    ).hexdigest()

    existing = (
        db.query(PickGapAssessment).filter_by(request_hash=request_hash).one_or_none()
    )
    if existing is not None:
        db.commit()
        return assessment_payload(db, existing, created=False), False

    entries = (
        db.query(CatalogEntry)
        .filter(
            CatalogEntry.compilation_id == compilation.id,
            CatalogEntry.status == "published",
        )
        .order_by(CatalogEntry.origin_time, CatalogEntry.id)
        .all()
    )
    if not entries:
        raise ApiError(
            422,
            "no_published_entries",
            f"catalog compilation {compilation.id} has no published entries; "
            "publish at least one entry before assessing P-pick gaps",
        )

    event_rows, frozen_entries = _evaluate_entries(db, entries, req.arrival_tolerance_s)
    station_rows = _aggregate_stations(event_rows, req.min_observable_events)

    assessment = PickGapAssessment(
        request_hash=request_hash,
        request=canon,
        compilation_id=compilation.id,
        arrival_tolerance_s=float(req.arrival_tolerance_s),
        min_observable_events=int(req.min_observable_events),
        frozen={
            "compilation_id": compilation.id,
            "selection": {"entry_status": "published"},
            "parameters": {
                "arrival_tolerance_s": float(req.arrival_tolerance_s),
                "min_observable_events": int(req.min_observable_events),
            },
            "entries": frozen_entries,
        },
        entry_revisions={e["entry_id"]: e["revision"] for e in frozen_entries},
        published_entry_count=len(entries),
        event_count=len(frozen_entries),
        station_count=len(station_rows),
        observable_count=sum(s["observable_count"] for s in station_rows),
        hit_count=sum(s["hit_count"] for s in station_rows),
        gap_count=sum(s["gap_count"] for s in station_rows),
    )
    db.add(assessment)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving assessment.
        db.rollback()
        winner = db.query(PickGapAssessment).filter_by(request_hash=request_hash).one()
        return assessment_payload(db, winner, created=False), False

    for idx, s in enumerate(station_rows):
        db.add(
            PickGapStation(
                assessment_id=assessment.id,
                station=s["station"],
                sort_index=idx,
                observable_count=s["observable_count"],
                hit_count=s["hit_count"],
                gap_count=s["gap_count"],
                gap_rate=s["gap_rate"],
                evidence_sufficient=1 if s["evidence_sufficient"] else 0,
            )
        )
    for idx, row in enumerate(event_rows):
        db.add(
            PickGapEvent(
                assessment_id=assessment.id,
                sort_index=idx,
                entry_id=row["entry_id"],
                revision=row["revision"],
                run_id=row["run_id"],
                event_id=row["event_id"],
                station=row["station"],
                status=row["status"],
                outcome=row["outcome"],
                detail=row["detail"],
            )
        )
    db.commit()
    db.refresh(assessment)
    return assessment_payload(db, assessment, created=True), True


def canonical_request(req: PickGapCreate) -> dict:
    return {
        "compilation_id": req.compilation_id,
        "arrival_tolerance_s": float(req.arrival_tolerance_s),
        "min_observable_events": int(req.min_observable_events),
    }


# --------------------------------------------------------------------------- #
# per-entry evaluation
# --------------------------------------------------------------------------- #
def _evaluate_entries(
    db: Session, entries: list[CatalogEntry], tolerance_s: float
) -> tuple[list[dict], list[dict]]:
    """Return (flat evidence rows, frozen per-entry context)."""
    rows: list[dict] = []
    frozen_entries: list[dict] = []
    for entry in entries:
        run = db.get(Run, entry.master_run_id)
        event = db.get(Event, entry.master_event_id)
        if run is None:  # pragma: no cover - FK integrity
            raise ApiError(404, "run_not_found", f"run {entry.master_run_id} not found")
        if event is None:  # pragma: no cover - FK integrity
            raise ApiError(
                404, "event_not_found", f"event {entry.master_event_id} not found"
            )

        model_row = db.get(VelocityModel, run.model_id)
        layers = canonical_layers(model_row.layers)
        model = LayeredModel.from_dicts(layers)

        cal_row = None
        p_offsets: dict[str, float] = {}
        if run.calibration_id is not None:
            cal_row = db.get(StationCalibration, run.calibration_id)
            terms = db.query(CalibrationTerm).filter_by(
                calibration_id=cal_row.id, phase="P"
            ).all()
            p_offsets = {t.station: float(t.offset_s) for t in terms}

        waveforms = (
            db.query(Waveform)
            .filter_by(ingestion_id=run.ingestion_id)
            .all()
        )
        picks = db.query(Pick).filter_by(ingestion_id=run.ingestion_id).all()

        origin_epoch = to_epoch(event.origin_time)
        for wf in sorted(waveforms, key=lambda w: w.station):
            rows.append(
                _evaluate_station(
                    entry, run, event, model_row, cal_row, model, p_offsets,
                    wf, picks, origin_epoch, tolerance_s,
                )
            )

        frozen_entries.append(
            {
                "entry_id": entry.id,
                "revision": entry.revision,
                "status": entry.status,
                "origin_time": iso_z(entry.origin_time),
                "run_id": run.id,
                "event_id": event.id,
                "ingestion_id": run.ingestion_id,
                "model": {"name": model_row.name, "version": model_row.version,
                          "layers": layers},
                "calibration": None
                if cal_row is None
                else {
                    "calibration_id": cal_row.id,
                    "version": cal_row.calibration_version,
                    "status": cal_row.status,
                    "p_terms": [
                        {"station": code, "offset_s": round(p_offsets[code], 6)}
                        for code in sorted(p_offsets)
                    ],
                },
            }
        )

    rows.sort(key=lambda r: (r["_order"], r["station"]))
    frozen_entries.sort(key=lambda e: (e["origin_time"], e["entry_id"]))
    return rows, frozen_entries


def _waveform_end_epoch(wf: Waveform) -> float:
    """Epoch of the waveform's last sample (ObsPy convention: npts-1 samples)."""
    return to_epoch(wf.start_time) + (wf.npts - 1) / float(wf.sampling_rate)


def _evaluate_station(
    entry, run, event, model_row, cal_row, model, p_offsets,
    wf: Waveform, picks: list[Pick], origin_epoch: float, tolerance_s: float,
) -> dict:
    dist = distance_km(event.latitude, event.longitude, wf.latitude, wf.longitude)
    travel_time_s, _ray_p = model.travel_time(
        event.depth_km + wf.elevation_m / 1000.0, dist, "P"
    )
    p_offset_s = float(p_offsets.get(wf.station, 0.0))
    pred_epoch = origin_epoch + travel_time_s + p_offset_s
    win_start, win_end = pred_epoch - tolerance_s, pred_epoch + tolerance_s

    wf_start = to_epoch(wf.start_time)
    wf_end = _waveform_end_epoch(wf)
    covered = wf_start <= win_start and wf_end >= win_end

    detail = {
        "entry_id": entry.id,
        "revision": entry.revision,
        "origin_time": iso_z(event.origin_time),
        "run_id": run.id,
        "event_id": event.id,
        "ingestion_id": run.ingestion_id,
        "station": wf.station,
        "station_coordinates": {
            "latitude": wf.latitude,
            "longitude": wf.longitude,
            "elevation_m": wf.elevation_m,
        },
        "model": {"name": model_row.name, "version": model_row.version},
        "calibration": (
            None
            if cal_row is None
            else {"version": cal_row.calibration_version, "status": cal_row.status}
        ),
        "p_calibration_offset_s": round(p_offset_s, 6),
        "distance_km": round(dist, 3),
        "travel_time_p_s": round(travel_time_s, 6),
        "predicted_p_time": iso_z(from_epoch(pred_epoch)),
        "window": {
            "start": iso_z(from_epoch(win_start)),
            "end": iso_z(from_epoch(win_end)),
            "tolerance_s": float(tolerance_s),
        },
        "coverage": {
            "start": iso_z(wf.start_time),
            "end": iso_z(from_epoch(wf_end)),
            "npts": wf.npts,
            "sampling_rate": wf.sampling_rate,
        },
    }

    order_key = (entry.origin_time, entry.id)
    if not covered:
        detail.update(
            {
                "observable": False,
                "unobservable_reason": REASON_WAVEFORM_NOT_COVERED,
                "matching_picks": [],
            }
        )
        return {
            "entry_id": entry.id,
            "revision": entry.revision,
            "run_id": run.id,
            "event_id": event.id,
            "station": wf.station,
            "status": STATUS_UNOBSERVABLE,
            "outcome": None,
            "detail": detail,
            "_order": order_key,
        }

    matching, excluded = _window_picks(picks, wf.station, win_start, win_end)
    outcome = OUTCOME_HIT if matching else OUTCOME_GAP
    detail.update(
        {
            "observable": True,
            "unobservable_reason": None,
            "matching_picks": matching,
            "excluded_picks": excluded,
        }
    )
    return {
        "entry_id": entry.id,
        "revision": entry.revision,
        "run_id": run.id,
        "event_id": event.id,
        "station": wf.station,
        "status": STATUS_OBSERVABLE,
        "outcome": outcome,
        "detail": detail,
        "_order": order_key,
    }


def _is_p_candidate(pick: Pick) -> bool:
    """Manual P picks and STA/LTA candidates count; manual S picks never do."""
    if pick.source == "stalta":
        return True
    return pick.source == "manual" and pick.phase == "P"


def _window_picks(
    picks: list[Pick], station: str, win_start: float, win_end: float
) -> tuple[list[dict], list[dict]]:
    """Partition the ingestion's in-window picks for this station.

    Matching P evidence (manual P / STA/LTA) is returned separately from
    excluded picks (manual S picks inside the window), so the report makes the
    "manual S does not count" decision auditable.
    """
    matching: list[dict] = []
    excluded: list[dict] = []
    for p in picks:
        if p.station != station:
            continue
        t = to_epoch(p.time_utc)
        if t < win_start or t > win_end:
            continue
        payload = {
            "pick_id": p.id,
            "time": iso_z(p.time_utc),
            "phase": p.phase,
            "source": p.source,
            "weight": round(p.weight, 3),
            "snr": round(p.snr, 4) if p.snr is not None else None,
            "delta_s": round(t - ((win_start + win_end) / 2.0), 6),
        }
        if _is_p_candidate(p):
            matching.append(payload)
        else:
            excluded.append(payload)
    matching.sort(key=lambda q: (q["time"], q["source"], q["pick_id"]))
    excluded.sort(key=lambda q: (q["time"], q["source"], q["pick_id"]))
    return matching, excluded


# --------------------------------------------------------------------------- #
# per-station aggregation
# --------------------------------------------------------------------------- #
def _aggregate_stations(event_rows: list[dict], min_observable: int) -> list[dict]:
    by_station: dict[str, dict] = {}
    for row in event_rows:
        agg = by_station.setdefault(
            row["station"],
            {"station": row["station"], "observable_count": 0,
             "hit_count": 0, "gap_count": 0},
        )
        if row["status"] != STATUS_OBSERVABLE:
            continue
        agg["observable_count"] += 1
        if row["outcome"] == OUTCOME_HIT:
            agg["hit_count"] += 1
        else:
            agg["gap_count"] += 1

    for agg in by_station.values():
        sufficient = agg["observable_count"] >= min_observable
        agg["evidence_sufficient"] = sufficient
        agg["gap_rate"] = (
            round(agg["gap_count"] / agg["observable_count"], 6)
            if sufficient and agg["observable_count"]
            else None
        )

    # Stable presentation order: scored stations first by descending gap rate
    # (then station code), evidence-insufficient stations after, by station.
    return sorted(
        by_station.values(),
        key=lambda s: (
            0 if s["evidence_sufficient"] else 1,
            -(s["gap_rate"] if s["gap_rate"] is not None else -1.0),
            s["station"],
        ),
    )


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_assessment(db: Session, assessment_id: str) -> dict:
    row = db.get(PickGapAssessment, assessment_id)
    if row is None:
        raise ApiError(
            404,
            "pick_gap_assessment_not_found",
            f"pick gap assessment {assessment_id} not found",
        )
    return assessment_payload(db, row, created=False)


def list_assessments(
    db: Session, compilation_id: str | None, limit: int, cursor: str | None
) -> dict:
    if compilation_id is not None and db.get(CatalogCompilation, compilation_id) is None:
        raise ApiError(
            404,
            "catalog_compilation_not_found",
            f"catalog compilation {compilation_id} not found",
        )

    filters = {"compilation_id": compilation_id, "limit": limit}
    fingerprint = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]

    cursor_at, cursor_id = None, None
    if cursor is not None:
        cursor_at, cursor_id = _decode_list_cursor(cursor, fingerprint)

    q = db.query(PickGapAssessment)
    if compilation_id is not None:
        q = q.filter(PickGapAssessment.compilation_id == compilation_id)
    if cursor_at is not None:
        q = q.filter(
            (PickGapAssessment.created_at > cursor_at)
            | (
                (PickGapAssessment.created_at == cursor_at)
                & (PickGapAssessment.id > cursor_id)
            )
        )
    q = q.order_by(PickGapAssessment.created_at, PickGapAssessment.id).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_list_cursor(last.created_at, last.id, fingerprint)
    return {
        "assessments": [assessment_summary(a) for a in page],
        "next_cursor": next_cursor,
    }


def list_events(
    db: Session,
    assessment_id: str,
    status: str,
    limit: int,
    cursor: str | None,
) -> dict:
    assessment = db.get(PickGapAssessment, assessment_id)
    if assessment is None:
        raise ApiError(
            404,
            "pick_gap_assessment_not_found",
            f"pick gap assessment {assessment_id} not found",
        )

    fingerprint = hashlib.sha256(
        json.dumps({"assessment_id": assessment.id, "status": status}).encode()
    ).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_event_cursor(cursor, fingerprint)

    q = db.query(PickGapEvent).filter_by(assessment_id=assessment.id)
    if status != "all":
        q = q.filter(PickGapEvent.status == status)
    if cursor_index is not None:
        q = q.filter(PickGapEvent.sort_index > cursor_index)
    q = q.order_by(PickGapEvent.sort_index).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_event_cursor(page[-1].sort_index, fingerprint)
    return {
        "assessment_id": assessment.id,
        "compilation_id": assessment.compilation_id,
        "event_count": assessment.event_count,
        "status": status,
        "events": [event_payload(r) for r in page],
        "next_cursor": next_cursor,
    }


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def _encode_list_cursor(created_at: datetime, assessment_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {"t": created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": assessment_id,
         "fp": fingerprint}
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_list_cursor(cursor: str, fingerprint: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_at = datetime.strptime(payload["t"], "%Y-%m-%dT%H:%M:%S.%f")
        token_id = payload["id"]
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422, "invalid_cursor",
            "cursor was issued for a different assessment query "
            "(compilation filter mismatch)",
        )
    return token_at, token_id


def _encode_event_cursor(sort_index: int, fingerprint: str) -> str:
    raw = json.dumps({"i": sort_index, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_event_cursor(cursor: str, fingerprint: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_i = int(payload["i"])
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422, "invalid_cursor",
            "cursor was issued for a different assessment or status filter",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def station_payload(row: PickGapStation) -> dict:
    return {
        "station": row.station,
        "sort_index": row.sort_index,
        "observable_count": row.observable_count,
        "hit_count": row.hit_count,
        "gap_count": row.gap_count,
        "gap_rate": row.gap_rate,
        "evidence_sufficient": bool(row.evidence_sufficient),
    }


def event_payload(row: PickGapEvent) -> dict:
    return {
        "sort_index": row.sort_index,
        "entry_id": row.entry_id,
        "revision": row.revision,
        "run_id": row.run_id,
        "event_id": row.event_id,
        "station": row.station,
        "status": row.status,
        "outcome": row.outcome,
        **row.detail,
    }


def assessment_summary(a: PickGapAssessment) -> dict:
    return {
        "assessment_id": a.id,
        "compilation_id": a.compilation_id,
        "request": a.request,
        "arrival_tolerance_s": a.arrival_tolerance_s,
        "min_observable_events": a.min_observable_events,
        "published_entry_count": a.published_entry_count,
        "event_count": a.event_count,
        "station_count": a.station_count,
        "observable_count": a.observable_count,
        "hit_count": a.hit_count,
        "gap_count": a.gap_count,
        "created_at": iso_z(a.created_at),
    }


def assessment_payload(db: Session, a: PickGapAssessment, created: bool) -> dict:
    station_rows = (
        db.query(PickGapStation)
        .filter_by(assessment_id=a.id)
        .order_by(PickGapStation.sort_index)
        .all()
    )
    return {
        "assessment_id": a.id,
        "created": created,
        "compilation_id": a.compilation_id,
        "request": a.request,
        "arrival_tolerance_s": a.arrival_tolerance_s,
        "min_observable_events": a.min_observable_events,
        "published_entry_count": a.published_entry_count,
        "event_count": a.event_count,
        "station_count": a.station_count,
        "observable_count": a.observable_count,
        "hit_count": a.hit_count,
        "gap_count": a.gap_count,
        "entry_revisions": dict(a.entry_revisions),
        "frozen": a.frozen,
        "stations": [station_payload(s) for s in station_rows],
        "created_at": iso_z(a.created_at),
    }
