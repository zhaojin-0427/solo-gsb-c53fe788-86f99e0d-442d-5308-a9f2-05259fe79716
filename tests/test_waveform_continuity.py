"""End-to-end tests for the station waveform acquisition continuity API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import make_waveform_b64, station_ring

SR = 100.0
BASE_DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
TARGET = "CW01"
T_LAT, T_LON, T_ELEV = 35.25, -118.0, 0.0
ENDPOINT = f"/v1/stations/{TARGET}/waveform-continuity"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _waveform(code, start, duration, *, arrivals=(), noise=0.02,
              lat=T_LAT, lon=T_LON, elev=T_ELEV, sr=SR):
    b64 = make_waveform_b64(code, sr, _iso(start), arrivals, duration=duration,
                            noise=noise)
    return {
        "station": code, "latitude": lat, "longitude": lon,
        "elevation_m": elev, "sampling_rate": sr,
        "start_time": _iso(start), "data_b64": b64,
    }


def _ingest(client, target_start, target_dur, filler_code, *,
            target_arrivals=(), filler_start=None, filler_dur=None,
            filler_arrivals=(), target_noise=0.02):
    """Create one ingestion containing TARGET plus a unique filler station."""
    body = {
        "waveforms": [
            _waveform(TARGET, target_start, target_dur,
                      arrivals=target_arrivals, noise=target_noise),
            _waveform(filler_code, filler_start or target_start,
                      filler_dur or target_dur, arrivals=filler_arrivals,
                      lat=35.30, noise=0.05),
        ],
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 2},
    }
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["ingestion_id"]


def _check(client, station=TARGET, start=BASE_DT, end=BASE_DT + timedelta(seconds=60),
           **extra):
    params = {"start_time": _iso(start), "end_time": _iso(end), **extra}
    return client.get(f"/v1/stations/{station}/waveform-continuity", params=params)


# --------------------------------------------------------------------------- #
# validation / not-found semantics
# --------------------------------------------------------------------------- #
def test_unknown_station_returns_404(client):
    _ingest(client, BASE_DT, 20.0, "FF01")
    resp = _check(client, station="NOPE")
    assert resp.status_code == 404
    err = resp.json()["detail"]
    assert err["code"] == "station_not_found"


def test_empty_or_inverted_window_rejected(client):
    _ingest(client, BASE_DT, 20.0, "FF01")
    for s, e in [
        (_iso(BASE_DT), _iso(BASE_DT)),              # empty
        (_iso(BASE_DT + timedelta(seconds=10)), _iso(BASE_DT)),  # inverted
    ]:
        resp = client.get(ENDPOINT, params={"start_time": s, "end_time": e})
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["code"] == "invalid_time_window"


def test_missing_or_unparseable_bound_rejected(client):
    _ingest(client, BASE_DT, 20.0, "FF01")
    resp = client.get(ENDPOINT, params={"start_time": _iso(BASE_DT)})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "invalid_time_window"

    resp = client.get(
        ENDPOINT,
        params={"start_time": "not-a-time", "end_time": _iso(BASE_DT)},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "invalid_time_window"


def test_known_station_with_no_record_in_window_returns_whole_gap(client):
    _ingest(client, BASE_DT + timedelta(seconds=100), 20.0, "FF01")
    resp = _check(client)
    assert resp.status_code == 200, resp.text
    rep = resp.json()
    assert rep["coverage"] == []
    assert rep["overlaps"] == []
    assert len(rep["gaps"]) == 1
    gap = rep["gaps"][0]
    assert gap["gap_kind"] == "whole"
    assert gap["start"] == _iso(BASE_DT)
    assert gap["end"] == _iso(BASE_DT + timedelta(seconds=60))
    assert gap["duration_s"] == 60.0
    assert rep["totals"]["gap_s"] == 60.0
    assert rep["totals"]["covered_s"] == 0.0


# --------------------------------------------------------------------------- #
# coverage / leading+trailing gaps
# --------------------------------------------------------------------------- #
def test_single_segment_fully_inside_window_reports_leading_trailing_gaps(client):
    _ingest(client, BASE_DT + timedelta(seconds=10), 20.0, "FF01")
    resp = _check(client)
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT + timedelta(seconds=10))
    # half-open: start + npts/sr, npts = 20*100 = 2000 -> end at 30 s
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=30))
    assert cov["duration_s"] == 20.0
    assert cov["coverage_count"] == 1
    assert len(cov["ingestion_ids"]) == 1

    kinds = [(g["gap_kind"], g["duration_s"]) for g in rep["gaps"]]
    assert kinds == [("leading", 10.0), ("trailing", 30.0)]

    totals = rep["totals"]
    assert totals["covered_s"] == 20.0
    assert totals["gap_s"] == 40.0
    assert totals["overlap_s"] == 0.0
    assert totals["window_s"] == 60.0
    assert totals["covered_segment_count"] == 1
    assert totals["gap_count"] == 2
    assert totals["overlap_segment_count"] == 0


def test_segments_are_clipped_to_window_half_open(client):
    # Segment covers [ -10, 20 ); clipped to the window [0, 60) -> [0, 20).
    _ingest(client, BASE_DT - timedelta(seconds=10), 30.0, "FF01")
    resp = _check(client)
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert cov["duration_s"] == 20.0
    assert rep["gaps"][0]["gap_kind"] == "trailing"
    assert rep["gaps"][0]["duration_s"] == 40.0
    assert rep["gaps"][0]["start"] == _iso(BASE_DT + timedelta(seconds=20))

    # End exactly at the window end: covered because the request interval is
    # half-open; the segment's exclusive end may equal end_time.
    resp = _check(client, end=BASE_DT + timedelta(seconds=20))
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    assert rep["coverage"][0]["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert rep["gaps"] == []


def test_segment_touching_window_only_at_endpoint_is_a_full_gap(client):
    # Segment ends exactly at the window start: zero-width half-open overlap.
    _ingest(client, BASE_DT - timedelta(seconds=20), 20.0, "FF01")
    resp = _check(client)
    rep = resp.json()
    assert rep["coverage"] == []
    assert len(rep["gaps"]) == 1
    assert rep["gaps"][0]["gap_kind"] == "whole"


# --------------------------------------------------------------------------- #
# abutting segments merge; internal gaps
# --------------------------------------------------------------------------- #
def test_abutting_unique_segments_merge_into_one_covered_interval(client):
    id1 = _ingest(client, BASE_DT, 20.0, "FF01")
    id2 = _ingest(client, BASE_DT + timedelta(seconds=20), 20.0, "FF02")
    resp = _check(client)
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=40))
    assert cov["duration_s"] == 40.0
    # A merged covered interval still carries every contributing ingestion.
    assert sorted(cov["ingestion_ids"]) == sorted([id1, id2])
    assert len(rep["gaps"]) == 1
    assert rep["gaps"][0]["gap_kind"] == "trailing"


def test_internal_gap_is_classified_and_durations_add_up(client):
    _ingest(client, BASE_DT, 10.0, "FF01")
    _ingest(client, BASE_DT + timedelta(seconds=20), 10.0, "FF02")
    resp = _check(client)
    rep = resp.json()
    assert len(rep["coverage"]) == 2
    assert [c["duration_s"] for c in rep["coverage"]] == [10.0, 10.0]
    assert len(rep["gaps"]) == 2
    assert [g["gap_kind"] for g in rep["gaps"]] == ["internal", "trailing"]
    internal = rep["gaps"][0]
    assert internal["start"] == _iso(BASE_DT + timedelta(seconds=10))
    assert internal["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert internal["duration_s"] == 10.0
    assert rep["totals"]["gap_s"] == 40.0


# --------------------------------------------------------------------------- #
# duplicates (same start / sampling rate / npts / data hash)
# --------------------------------------------------------------------------- #
def test_identical_waveforms_across_ingestions_count_once_keep_all_ids(client):
    # Identical code + start + duration + deterministic RNG noise => identical
    # data bytes and hash; different filler stations => different ingestions.
    id1 = _ingest(client, BASE_DT, 20.0, "FF01")
    id2 = _ingest(client, BASE_DT, 20.0, "FF02")
    id3 = _ingest(client, BASE_DT, 20.0, "FF03")

    resp = _check(client)
    rep = resp.json()
    assert rep["ingested_waveform_count"] == 3
    assert rep["unique_segment_count"] == 1
    assert rep["duplicate_waveform_count"] == 1
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["coverage_count"] == 1  # one unique segment, not three
    assert sorted(cov["ingestion_ids"]) == sorted([id1, id2, id3])

    dup = rep["duplicates"][0]
    assert dup["npts"] == 2000
    assert dup["sampling_rate"] == 100.0
    assert dup["ingestion_count"] == 3
    assert sorted(dup["ingestion_ids"]) == sorted([id1, id2, id3])
    assert len(dup["data_sha256"]) == 64


def test_different_data_hash_is_a_separate_segment_and_overlaps(client):
    # Same range, different sample content (different RNG stream via arrivals):
    # not a duplicate -> two unique segments overlapping over the full range.
    id1 = _ingest(client, BASE_DT, 20.0, "FF01")
    id2 = _ingest(
        client, BASE_DT, 20.0, "FF02",
        target_arrivals=[(5.0, 4.0)], target_noise=0.09,
    )
    resp = _check(client)
    rep = resp.json()
    assert rep["duplicate_waveform_count"] == 0
    assert rep["unique_segment_count"] == 2
    # The depth->=1 union stays in coverage: one merged coverage segment spans
    # the full acquired range and reports the peak depth (2); the multiply
    # covered interior is *additionally* listed under overlaps.
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert cov["duration_s"] == 20.0
    assert cov["coverage_count"] == 2  # peak unique-segment depth in-segment
    assert sorted(cov["ingestion_ids"]) == sorted([id1, id2])
    assert len(rep["overlaps"]) == 1
    ov = rep["overlaps"][0]
    assert ov["start"] == _iso(BASE_DT)
    assert ov["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert ov["coverage_count"] == 2
    assert sorted(ov["ingestion_ids"]) == sorted([id1, id2])
    totals = rep["totals"]
    assert totals["covered_s"] == 20.0   # overlap time is acquired union time
    assert totals["gap_s"] == 40.0
    assert totals["overlap_s"] == 20.0   # only the multiply-covered interior
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]
    assert totals["covered_segment_count"] == 1
    assert totals["overlap_segment_count"] == 1
    # Explicitly: overlap says nothing about sample-content conflicts.
    assert rep["overlap_semantics"] == "acquisition_range_only"


def test_partial_overlap_merges_into_one_coverage_segment(client):
    # [0, 30) and [20, 50): depth 1 on [0,20), depth 2 on [20,30), depth 1 on
    # [30,50).  The depth->=1 union is one abutting coverage segment
    # [0,50) with peak depth 2; the [20,30) interior is separately the overlap.
    id1 = _ingest(client, BASE_DT, 30.0, "FF01")
    id2 = _ingest(client, BASE_DT + timedelta(seconds=20), 30.0, "FF02")
    resp = _check(client)
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=50))
    assert cov["duration_s"] == 50.0
    assert cov["coverage_count"] == 2  # peak depth, even at the depth-1 ends
    assert sorted(cov["ingestion_ids"]) == sorted([id1, id2])
    assert [o["duration_s"] for o in rep["overlaps"]] == [10.0]
    ov = rep["overlaps"][0]
    assert ov["start"] == _iso(BASE_DT + timedelta(seconds=20))
    assert ov["end"] == _iso(BASE_DT + timedelta(seconds=30))
    assert sorted(ov["ingestion_ids"]) == sorted([id1, id2])
    # Gaps: none leading/internal, one trailing.
    assert [g["gap_kind"] for g in rep["gaps"]] == ["trailing"]
    totals = rep["totals"]
    assert totals["covered_s"] == 50.0
    assert totals["overlap_s"] == 10.0
    assert totals["gap_s"] == 10.0
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]
    assert totals["covered_segment_count"] == 1
    assert totals["overlap_segment_count"] == 1


def test_three_deep_overlap_merges_same_class_abutments(client):
    # [0, 30), [10, 40), [20, 50):
    # covered [0,10), overlap [10,20)(depth2), overlap [20,30)(depth3),
    # overlap [30,40)(depth2), covered [40,50), gap trailing.
    _ingest(client, BASE_DT, 30.0, "FF01")
    _ingest(client, BASE_DT + timedelta(seconds=10), 30.0, "FF02")
    _ingest(client, BASE_DT + timedelta(seconds=20), 30.0, "FF03")
    resp = _check(client)
    rep = resp.json()
    # Abutting depth-1/depth-2/depth-3 pieces form one coverage union [0,50)
    # carrying the peak depth 3 ...
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=50))
    assert cov["duration_s"] == 50.0
    assert cov["coverage_count"] == 3
    # ... while the separately listed overlap merges the abutting depth 2/3/2
    # pieces into [10,40) with peak depth 3.
    assert len(rep["overlaps"]) == 1
    assert rep["overlaps"][0]["start"] == _iso(BASE_DT + timedelta(seconds=10))
    assert rep["overlaps"][0]["end"] == _iso(BASE_DT + timedelta(seconds=40))
    assert rep["overlaps"][0]["duration_s"] == 30.0
    assert rep["overlaps"][0]["coverage_count"] == 3
    totals = rep["totals"]
    assert totals["covered_s"] == 50.0
    assert totals["overlap_s"] == 30.0
    assert totals["gap_s"] == 10.0
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]


def test_duplicate_plus_another_segment_overlap_uses_unique_depth(client):
    # Segment A duplicated in two ingestions (one unique segment); segment B
    # overlaps A in range.  Depth must count unique segments (2), not rows (3).
    id_a1 = _ingest(client, BASE_DT, 20.0, "FF01")
    id_a2 = _ingest(client, BASE_DT, 20.0, "FF02")
    id_b = _ingest(client, BASE_DT + timedelta(seconds=10), 20.0, "FF03",
                   target_arrivals=[(3.0, 5.0)], target_noise=0.11)
    resp = _check(client, end=BASE_DT + timedelta(seconds=40))
    rep = resp.json()
    assert rep["unique_segment_count"] == 2
    assert rep["duplicate_waveform_count"] == 1
    # One abutting coverage union [0,30), peak depth 2 and all three ingestion
    # ids; the [10,20) double-covered interior is the separate overlap.
    assert [c["duration_s"] for c in rep["coverage"]] == [30.0]
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=30))
    assert cov["coverage_count"] == 2
    assert sorted(cov["ingestion_ids"]) == sorted([id_a1, id_a2, id_b])
    assert [o["duration_s"] for o in rep["overlaps"]] == [10.0]
    ov = rep["overlaps"][0]
    assert ov["coverage_count"] == 2
    assert sorted(ov["ingestion_ids"]) == sorted([id_a1, id_a2, id_b])
    totals = rep["totals"]
    assert totals["covered_s"] == 30.0
    assert totals["overlap_s"] == 10.0
    assert totals["gap_s"] == 10.0
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]


def test_coverage_union_invariant_over_disjoint_overlaps(client):
    # A [0,10), B [5,15), C [40,50), D [45,55): two disjoint overlaps sitting
    # inside two disjoint coverage unions; overlap is counted once for overlap_s
    # but fully included in the acquired union covered_s.
    _ingest(client, BASE_DT, 10.0, "FF01")
    _ingest(client, BASE_DT + timedelta(seconds=5), 10.0, "FF02",
            target_arrivals=[(2.0, 4.0)], target_noise=0.07)
    _ingest(client, BASE_DT + timedelta(seconds=40), 10.0, "FF03")
    _ingest(client, BASE_DT + timedelta(seconds=45), 10.0, "FF04",
            target_arrivals=[(2.0, 4.0)], target_noise=0.07)
    resp = _check(client)
    rep = resp.json()

    assert len(rep["coverage"]) == 2
    assert [c["duration_s"] for c in rep["coverage"]] == [15.0, 15.0]
    assert [c["coverage_count"] for c in rep["coverage"]] == [2, 2]
    assert len(rep["overlaps"]) == 2
    assert [o["duration_s"] for o in rep["overlaps"]] == [5.0, 5.0]

    totals = rep["totals"]
    assert totals["covered_s"] == 30.0
    assert totals["gap_s"] == 30.0
    assert totals["overlap_s"] == 10.0
    assert totals["window_s"] == 60.0
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]
    assert totals["covered_segment_count"] == 2
    assert totals["overlap_segment_count"] == 2

    # Every overlap piece must lie strictly inside a coverage union segment.
    for ov in rep["overlaps"]:
        assert any(
            cov["start"] <= ov["start"] and ov["end"] <= cov["end"]
            for cov in rep["coverage"]
        )

    # Gaps complement the coverage union exactly: 25 s internal + 5 s trailing.
    assert [(g["gap_kind"], g["duration_s"]) for g in rep["gaps"]] == [
        ("internal", 25.0), ("trailing", 5.0),
    ]


def test_overlap_at_window_edges_is_inside_coverage_union(client):
    # Two different-hash segments covering [0,20) overlap over the window's
    # leading 10 s (segments [-10,10) and [0,20) clipped): the union must still
    # start exactly at the window start with no leading gap.
    _ingest(client, BASE_DT - timedelta(seconds=10), 20.0, "FF01")
    _ingest(client, BASE_DT, 20.0, "FF02",
            target_arrivals=[(4.0, 4.0)], target_noise=0.07)
    resp = _check(client)
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == _iso(BASE_DT)
    assert cov["end"] == _iso(BASE_DT + timedelta(seconds=20))
    assert cov["coverage_count"] == 2
    assert [o["start"] for o in rep["overlaps"]] == [_iso(BASE_DT)]
    totals = rep["totals"]
    assert totals["covered_s"] == 20.0
    assert totals["overlap_s"] == 10.0
    assert totals["gap_s"] == 40.0
    assert totals["covered_s"] + totals["gap_s"] == totals["window_s"]


# --------------------------------------------------------------------------- #
# independence: reads across all ingestions; read-only; repeatable
# --------------------------------------------------------------------------- #
def test_report_reads_across_ingestions_and_is_repeatable(client):
    _ingest(client, BASE_DT, 10.0, "FF01")
    _ingest(client, BASE_DT + timedelta(seconds=30), 10.0, "FF02")
    first = _check(client)
    second = _check(client)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()

    # Window outside every ingestion: known station, whole gap, still 200.
    resp = _check(client, start=BASE_DT + timedelta(hours=2),
                  end=BASE_DT + timedelta(hours=3))
    assert resp.status_code == 200
    rep = resp.json()
    assert rep["ingested_waveform_count"] == 2
    assert rep["unique_segment_count"] == 0
    assert len(rep["gaps"]) == 1
    assert rep["gaps"][0]["gap_kind"] == "whole"


def test_continuity_check_does_not_mutate_ingestions_or_pick_gaps(client):
    # Build a pick-gap style ingestion (single event) and confirm the
    # continuity endpoint neither changes process output nor needs runs.
    data = client.post(
        "/v1/process",
        json={
            "waveforms": [
                _waveform(TARGET, BASE_DT, 60.0,
                          arrivals=[(34.0, 4.0)], noise=0.02),
                _waveform("FF99", BASE_DT, 60.0, lat=35.30,
                          arrivals=[(34.0, 4.0)], noise=0.02),
            ],
            "manual_picks": [],
            "model": {"name": "default", "version": "1.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 2},
        },
    ).json()
    ingestion_id = data["ingestion_id"]

    for _ in range(2):
        rep = _check(
            client,
            start=BASE_DT + timedelta(seconds=20),
            end=BASE_DT + timedelta(seconds=50),
        ).json()
        assert len(rep["coverage"]) == 1
        assert rep["coverage"][0]["ingestion_ids"] == [ingestion_id]
        assert rep["gaps"] == []


def test_window_endpoints_are_reported_as_requested(client):
    _ingest(client, BASE_DT + timedelta(seconds=5), 10.0, "FF01")
    resp = _check(client, start=BASE_DT + timedelta(seconds=3),
                  end=BASE_DT + timedelta(seconds=33))
    rep = resp.json()
    assert rep["window"] == {
        "start": _iso(BASE_DT + timedelta(seconds=3)),
        "end": _iso(BASE_DT + timedelta(seconds=33)),
        "interval": "half_open",
    }
    # clipped coverage starts/ends on the segment, gaps on the window bounds.
    assert rep["gaps"][0]["start"] == _iso(BASE_DT + timedelta(seconds=3))
    assert rep["gaps"][-1]["end"] == _iso(BASE_DT + timedelta(seconds=33))
