"""End-to-end tests for the station P-wave pick gap assessment."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from tests.conftest import make_waveform_b64, station_ring

from app.config import DEFAULT_MODEL_LAYERS
from app.services.association import distance_km
from app.services.traveltime import LayeredModel
from app.services.pipeline import to_epoch

SR = 100.0
DURATION = 120.0
ORIGIN_OFFSET = 30.0
MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
BASE_DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
TOL = 2.0


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _waveform(code, lat, lon, elev, arrivals, *, start=None, duration=DURATION,
              noise=0.02):
    start = start or BASE_DT
    b64 = make_waveform_b64(code, SR, _iso(start), arrivals, duration=duration,
                            noise=noise)
    return {
        "station": code, "latitude": lat, "longitude": lon,
        "elevation_m": elev, "sampling_rate": SR,
        "start_time": _iso(start), "data_b64": b64,
    }


def _event_window(
    prefix,
    *,
    n_stations=8,
    quiet_stations=(),
    late_stations=(),
    manual_picks=None,
    ev_lat=35.05,
    ev_lon=-117.95,
    ev_depth=12.0,
    origin_offset=ORIGIN_OFFSET,
):
    """One /v1/process body; quiet/late stations carry no P pulse (gap/unobs)."""
    waveforms = []
    for code, lat, lon, elev in station_ring(prefix, n=n_stations):
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        z = ev_depth + elev / 1000.0
        if code in late_stations:
            # Window starts well after the predicted P arrival: the event is
            # still located by the other stations but this waveform cannot
            # cover the prediction.
            waveforms.append(
                _waveform(code, lat, lon, elev, [],
                          start=BASE_DT + timedelta(seconds=origin_offset + 10.0))
            )
            continue
        if code in quiet_stations:
            waveforms.append(_waveform(code, lat, lon, elev, []))
            continue
        tp = MODEL.travel_time(z, dist, "P")[0]
        ts = MODEL.travel_time(z, dist, "S")[0]
        waveforms.append(
            _waveform(code, lat, lon, elev,
                      [(origin_offset + tp, 4.0), (origin_offset + ts, 6.0)])
        )
    return {
        "waveforms": waveforms,
        "manual_picks": manual_picks or [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }


def _process(client, body):
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["events"], "no event located"
    return data


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _publish_compilation(client, *, start=BASE_DT, days=30):
    resp = client.post(
        "/v1/catalog/compilations",
        json={
            "start_time": _iso(start),
            "end_time": _iso(start + timedelta(days=days)),
            "radius_km": 50.0, "origin_tolerance_s": 5.0,
            "depth_tolerance_km": 15.0, "min_quality": "C",
            "min_sources": 1, "min_shared_stations": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    comp = resp.json()
    assert comp["entry_count"] >= 1
    for entry in comp["entries"]:
        pub = client.post(
            f"/v1/catalog/entries/{entry['entry_id']}/publish",
            json={"expected_revision": 1},
        )
        assert pub.status_code == 200, pub.text
    return comp


def _assess(client, compilation_id, *, tolerance=TOL, min_obs=1):
    return client.post(
        "/v1/pick-gap-assessments",
        json={
            "compilation_id": compilation_id,
            "arrival_tolerance_s": tolerance,
            "min_observable_events": min_obs,
        },
    )


def _stations_by_code(rep):
    return {s["station"]: s for s in rep["stations"]}


# --------------------------------------------------------------------------- #
# basic scoring: hits, gaps, per-station aggregates
# --------------------------------------------------------------------------- #
def test_all_pulsed_stations_are_observable_hits(client):
    data = _process(client, _event_window("GH"))
    event = _best(data)
    comp = _publish_compilation(client)
    resp = _assess(client, comp["compilation_id"])
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["compilation_id"] == comp["compilation_id"]
    assert rep["published_entry_count"] == 1
    assert rep["event_count"] == 1
    assert rep["station_count"] == 8
    assert rep["observable_count"] == 8
    assert rep["hit_count"] == 8
    assert rep["gap_count"] == 0
    assert rep["arrival_tolerance_s"] == TOL
    assert rep["min_observable_events"] == 1

    for s in rep["stations"]:
        assert s["evidence_sufficient"] is True
        assert s["observable_count"] == 1
        assert s["hit_count"] == 1
        assert s["gap_count"] == 0
        assert s["gap_rate"] == 0.0

    # report echoes the frozen entry revision (publish bumped 1 -> 2)
    entry_id = comp["entries"][0]["entry_id"]
    assert rep["entry_revisions"] == {entry_id: 2}
    frozen_entry = rep["frozen"]["entries"][0]
    assert frozen_entry["entry_id"] == entry_id
    assert frozen_entry["revision"] == 2
    assert frozen_entry["run_id"] == data["run_id"]
    assert frozen_entry["event_id"] == event["event_id"]
    assert frozen_entry["model"]["name"] == "default"
    assert frozen_entry["model"]["version"] == "1.0"
    assert frozen_entry["calibration"] is None
    assert isinstance(frozen_entry["model"]["layers"], list)


def test_quiet_station_is_an_observable_gap(client):
    body = _event_window("GQ", quiet_stations={"GQ03"})
    _process(client, body)
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"]).json()

    assert rep["observable_count"] == 8  # quiet station still recorded a window
    assert rep["hit_count"] == 7
    assert rep["gap_count"] == 1
    s = _stations_by_code(rep)["GQ03"]
    assert s["observable_count"] == 1
    assert s["hit_count"] == 0
    assert s["gap_count"] == 1
    assert s["gap_rate"] == 1.0
    # gap stations sort ahead of zero-gap stations at equal evidence status
    assert rep["stations"][0]["station"] == "GQ03"

    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    row = next(r for r in events if r["station"] == "GQ03")
    assert row["status"] == "observable"
    assert row["outcome"] == "gap"
    assert row["observable"] is True
    assert row["unobservable_reason"] is None
    assert row["matching_picks"] == []
    assert row["window"]["tolerance_s"] == TOL


def test_manual_s_pick_inside_window_does_not_count(client):
    # No P pulse at one station, only a manual S label placed right on the
    # predicted P arrival: it must appear under excluded_picks, never as a hit.
    body = _event_window(
        "GS",
        quiet_stations={"GS05"},
        manual_picks=[
            {"station": "GS05", "phase": "S",
             "time": _iso(BASE_DT + timedelta(seconds=ORIGIN_OFFSET + 4.0)),
             "weight": 1.0}
        ],
    )
    _process(client, body)
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"], tolerance=5.0).json()

    s = _stations_by_code(rep)["GS05"]
    assert s["gap_count"] == 1 and s["hit_count"] == 0
    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    row = next(r for r in events if r["station"] == "GS05")
    assert row["outcome"] == "gap"
    assert row["matching_picks"] == []
    assert len(row["excluded_picks"]) == 1
    excluded = row["excluded_picks"][0]
    assert excluded["phase"] == "S"
    assert excluded["source"] == "manual"


def test_manual_p_pick_in_window_is_a_hit(client):
    # Quiet waveform (no detectable pulse) but a manual P pick exactly in the
    # predicted window counts as a hit.
    body = _event_window(
        "GP",
        quiet_stations={"GP02"},
        manual_picks=[
            {"station": "GP02", "phase": "P",
             "time": _iso(BASE_DT + timedelta(seconds=ORIGIN_OFFSET + 4.2)),
             "weight": 1.0}
        ],
    )
    _process(client, body)
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"], tolerance=5.0).json()
    s = _stations_by_code(rep)["GP02"]
    assert s["hit_count"] == 1 and s["gap_count"] == 0
    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    row = next(r for r in events if r["station"] == "GP02")
    assert row["outcome"] == "hit"
    assert len(row["matching_picks"]) == 1
    assert row["matching_picks"][0]["source"] == "manual"
    assert row["matching_picks"][0]["phase"] == "P"


def test_prediction_uses_origin_travel_time_and_window_evidence(client):
    data = _process(client, _event_window("GE"))
    event = _best(data)
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"], tolerance=1.5).json()
    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]

    origin = datetime.strptime(event["origin_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
    origin_t = to_epoch(origin)
    ring = {c: (lat, lon, elev) for c, lat, lon, elev in station_ring("GE")}
    epoch = datetime(1970, 1, 1)

    def _parse(iso):
        return (datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ") - epoch).total_seconds()

    for row in events:
        lat, lon, elev = ring[row["station"]]
        dist = distance_km(event["latitude"], event["longitude"], lat, lon)
        expect_tt = MODEL.travel_time(
            event["depth_km"] + elev / 1000.0, dist, "P"
        )[0]
        assert abs(row["travel_time_p_s"] - round(expect_tt, 6)) < 2e-3
        assert abs(row["distance_km"] - round(dist, 3)) <= 0.01
        assert row["p_calibration_offset_s"] == 0.0
        # predicted == origin (payload precision is ms) + travel time
        got_pred = _parse(row["predicted_p_time"])
        assert abs(got_pred - (origin_t + expect_tt)) < 2e-3
        # internal consistency of the frozen evidence window
        assert abs(_parse(row["window"]["start"]) - (got_pred - 1.5)) < 1e-9
        assert abs(_parse(row["window"]["end"]) - (got_pred + 1.5)) < 1e-9
        assert row["window"]["start"] < row["predicted_p_time"] < row["window"]["end"]
        # coverage record describes the original ingestion waveform
        assert row["coverage"]["npts"] == int(DURATION * SR)


# --------------------------------------------------------------------------- #
# observability gate: waveform must fully cover the tolerance window
# --------------------------------------------------------------------------- #
def test_shifted_window_is_unobservable_and_excluded_from_rates(client):
    body = _event_window("GU", late_stations={"GU06"})
    _process(client, body)
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"], min_obs=1).json()

    assert rep["observable_count"] == 7
    s = _stations_by_code(rep)["GU06"]
    assert s["observable_count"] == 0
    assert s["hit_count"] == 0 and s["gap_count"] == 0

    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    row = next(r for r in events if r["station"] == "GU06")
    assert row["status"] == "unobservable"
    assert row["outcome"] is None
    assert row["observable"] is False
    assert row["unobservable_reason"] == "waveform_not_covered"
    assert row["matching_picks"] == []

    filtered = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events",
        params={"status": "unobservable"},
    ).json()
    assert [r["station"] for r in filtered["events"]] == ["GU06"]
    obs = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events",
        params={"status": "observable"},
    ).json()
    assert len(obs["events"]) == 7


def test_evidence_insufficient_flag_and_null_rate(client):
    _process(client, _event_window("GI"))
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"], min_obs=2).json()
    assert rep["observable_count"] == 8
    for s in rep["stations"]:
        assert s["evidence_sufficient"] is False
        assert s["gap_rate"] is None
        assert s["observable_count"] == 1
    # evidence-insufficient stations follow scored stations and sort by code
    assert [s["station"] for s in rep["stations"]] == sorted(
        s["station"] for s in rep["stations"]
    )


# --------------------------------------------------------------------------- #
# published-only selection; entry counted once; frozen revisions
# --------------------------------------------------------------------------- #
def test_only_published_entries_count_each_once(client):
    # Event A
    _process(client, _event_window("GA"))
    comp_a = _publish_compilation(client)
    # Withdraw the entry of comp A
    entry_id = comp_a["entries"][0]["entry_id"]
    wd = client.post(
        f"/v1/catalog/entries/{entry_id}/withdraw",
        json={"expected_revision": 2, "reason": "duplicate"},
    )
    assert wd.status_code == 200, wd.text

    # Re-submitting while no entries are published is rejected without state.
    bad = _assess(client, comp_a["compilation_id"])
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "no_published_entries"

    # A later, distinct compilation over the same window: entry stays draft.
    draft = client.post(
        "/v1/catalog/compilations",
        json={
            "start_time": _iso(BASE_DT),
            "end_time": _iso(BASE_DT + timedelta(days=30)),
            "radius_km": 50.0, "origin_tolerance_s": 5.0,
            "depth_tolerance_km": 15.0, "min_quality": "C",
            "min_sources": 1, "min_shared_stations": 0,
        },
    )
    assert draft.status_code == 201
    bad2 = _assess(client, draft.json()["compilation_id"])
    assert bad2.status_code == 422
    assert bad2.json()["detail"]["code"] == "no_published_entries"


def test_two_published_entries_each_count_once(client):
    _process(client, _event_window("G1"))
    _process(
        client,
        _event_window(
            "G2", ev_lat=35.05, ev_lon=-118.4,
        ),
    )
    comp = _publish_compilation(client)
    assert comp["entry_count"] == 2
    rep = _assess(client, comp["compilation_id"]).json()
    assert rep["event_count"] == 2
    assert rep["published_entry_count"] == 2
    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events",
        params={"limit": 200},
    ).json()["events"]
    # one stable evidence row per (entry, station)
    assert len(events) == 16
    per_entry: dict[str, list[str]] = {}
    for r in events:
        per_entry.setdefault(r["entry_id"], []).append(r["station"])
    for stations in per_entry.values():
        assert sorted(stations) == stations
        assert len(stations) == 8
    # events ordered by catalog origin time then entry id then station
    keys = [(r["origin_time"], r["entry_id"], r["station"]) for r in events]
    assert keys == sorted(keys)


def test_report_freezes_revisions_and_survives_later_withdrawal(client):
    _process(client, _event_window("GF"))
    comp = _publish_compilation(client)
    entry_id = comp["entries"][0]["entry_id"]
    first = _assess(client, comp["compilation_id"], tolerance=1.0)
    assert first.status_code == 201
    rep = first.json()
    assert rep["entry_revisions"][entry_id] == 2

    wd = client.post(
        f"/v1/catalog/entries/{entry_id}/withdraw",
        json={"expected_revision": 2, "reason": "later retraction"},
    )
    assert wd.status_code == 200
    # repeat request after withdrawal: same frozen snapshot (revision 2)
    again = _assess(client, comp["compilation_id"], tolerance=1.0)
    assert again.status_code == 200
    assert again.json()["assessment_id"] == rep["assessment_id"]
    assert again.json()["entry_revisions"] == rep["entry_revisions"]
    assert again.json()["stations"] == rep["stations"]
    rows = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    assert all(r["revision"] == 2 for r in rows)


# --------------------------------------------------------------------------- #
# idempotency / concurrency / parameter identity
# --------------------------------------------------------------------------- #
def test_assessment_is_idempotent(client):
    _process(client, _event_window("GX"))
    comp = _publish_compilation(client)
    first = _assess(client, comp["compilation_id"])
    second = _assess(client, comp["compilation_id"])
    assert first.status_code == 201
    assert second.status_code == 200
    a, b = first.json(), second.json()
    assert a["assessment_id"] == b["assessment_id"]
    assert b["created"] is False
    assert a["stations"] == b["stations"]
    assert a["frozen"] == b["frozen"]


def test_different_parameters_are_distinct_assessments(client):
    _process(client, _event_window("GD"))
    comp = _publish_compilation(client)
    a = _assess(client, comp["compilation_id"], tolerance=1.0)
    b = _assess(client, comp["compilation_id"], tolerance=2.0)
    c = _assess(client, comp["compilation_id"], tolerance=1.0, min_obs=2)
    assert a.status_code == b.status_code == c.status_code == 201
    ids = {r.json()["assessment_id"] for r in (a, b, c)}
    assert len(ids) == 3


def test_concurrent_identical_assessments_collapse(client):
    _process(client, _event_window("GC"))
    comp = _publish_compilation(client)
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _assess(client, comp["compilation_id"], tolerance=3.0)
        results.append((r.status_code, r.json()["assessment_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def test_events_pagination_stable_cursor(client):
    _process(client, _event_window("G1"))
    _process(client, _event_window("G2", ev_lat=35.05, ev_lon=-118.4))
    comp = _publish_compilation(client)
    rep = _assess(client, comp["compilation_id"]).json()
    full = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events",
        params={"limit": 200},
    ).json()["events"]

    seen, cursor = [], None
    for _ in range(40):
        params = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        page = client.get(
            f"/v1/pick-gap-assessments/{rep['assessment_id']}/events",
            params=params,
        ).json()
        seen.extend(page["events"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert [r["sort_index"] for r in seen] == [r["sort_index"] for r in full]
    assert len(seen) == 16
    assert [r["sort_index"] for r in seen] == list(range(16))


def test_event_cursor_bound_to_assessment_and_status(client):
    _process(client, _event_window("GB"))
    comp1 = _publish_compilation(client)
    a1 = _assess(client, comp1["compilation_id"]).json()

    _process(client, _event_window("GB", ev_lat=35.9, ev_lon=-118.0))
    comp2 = _publish_compilation(
        client, start=BASE_DT - timedelta(days=10), days=60
    )
    a2 = _assess(client, comp2["compilation_id"]).json()

    page = client.get(
        f"/v1/pick-gap-assessments/{a1['assessment_id']}/events",
        params={"limit": 2},
    ).json()
    # cursor from a1 must not be accepted by a2
    bad = client.get(
        f"/v1/pick-gap-assessments/{a2['assessment_id']}/events",
        params={"limit": 2, "cursor": page["next_cursor"]},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"

    # a cursor minted for the observable filter is rejected under a different
    # status filter (fingerprint includes the status)
    obs_page = client.get(
        f"/v1/pick-gap-assessments/{a1['assessment_id']}/events",
        params={"limit": 2, "status": "observable"},
    ).json()
    mism = client.get(
        f"/v1/pick-gap-assessments/{a1['assessment_id']}/events",
        params={"limit": 2, "status": "all", "cursor": obs_page["next_cursor"]},
    )
    assert mism.status_code == 422
    assert mism.json()["detail"]["code"] == "invalid_cursor"


def test_assessment_list_pagination_and_filter(client):
    _process(client, _event_window("GL"))
    comp = _publish_compilation(client)
    a1 = _assess(client, comp["compilation_id"], tolerance=1.0).json()
    a2 = _assess(client, comp["compilation_id"], tolerance=2.0).json()

    by_comp = client.get(
        "/v1/pick-gap-assessments",
        params={"compilation_id": comp["compilation_id"]},
    ).json()
    assert {a["assessment_id"] for a in by_comp["assessments"]} == {
        a1["assessment_id"], a2["assessment_id"]
    }

    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/v1/pick-gap-assessments", params=params).json()
        seen.update(a["assessment_id"] for a in data["assessments"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert {a1["assessment_id"], a2["assessment_id"]} <= seen

    page = client.get(
        "/v1/pick-gap-assessments",
        params={"limit": 1, "compilation_id": comp["compilation_id"]},
    ).json()
    other_comp = client.post(
        "/v1/catalog/compilations",
        json={
            "start_time": _iso(BASE_DT),
            "end_time": _iso(BASE_DT + timedelta(days=30)),
            "radius_km": 50.0, "origin_tolerance_s": 5.0,
            "depth_tolerance_km": 15.0, "min_quality": "C",
            "min_sources": 1, "min_shared_stations": 0,
        },
    ).json()
    mism = client.get(
        "/v1/pick-gap-assessments",
        params={"limit": 1, "compilation_id": other_comp["compilation_id"],
                "cursor": page["next_cursor"]},
    )
    assert mism.status_code == 422
    assert mism.json()["detail"]["code"] == "invalid_cursor"


# --------------------------------------------------------------------------- #
# rejections / 404 / validation
# --------------------------------------------------------------------------- #
def test_unknown_compilation_404(client):
    resp = _assess(client, "0" * 32)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "catalog_compilation_not_found"


def test_unknown_assessment_404(client):
    r = client.get("/v1/pick-gap-assessments/deadbeef")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "pick_gap_assessment_not_found"
    r2 = client.get("/v1/pick-gap-assessments/deadbeef/events")
    assert r2.status_code == 404
    assert r2.json()["detail"]["code"] == "pick_gap_assessment_not_found"


def test_list_unknown_compilation_filter_404(client):
    r = client.get(
        "/v1/pick-gap-assessments", params={"compilation_id": "f" * 32}
    )
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "catalog_compilation_not_found"


def test_parameter_ranges_validated(client):
    body = {
        "compilation_id": "0" * 32,
        "arrival_tolerance_s": 0.05,
        "min_observable_events": 1,
    }
    r = client.post("/v1/pick-gap-assessments", json=body)
    assert r.status_code == 422
    body["arrival_tolerance_s"] = 10.5
    r = client.post("/v1/pick-gap-assessments", json=body)
    assert r.status_code == 422
    body["arrival_tolerance_s"] = 1.0
    body["min_observable_events"] = 101
    r = client.post("/v1/pick-gap-assessments", json=body)
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# read-only: catalog entries, runs, events never change
# --------------------------------------------------------------------------- #
def test_assessment_does_not_mutate_source_data(client):
    data = _process(client, _event_window("GR"))
    event = _best(data)
    comp_before = client.get(
        f"/v1/catalog/compilations/{_publish_compilation(client)['compilation_id']}"
    ).json()
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    cid = comp_before["compilation_id"]
    _assess(client, cid)
    _assess(client, cid)  # idempotent replay

    comp_after = client.get(f"/v1/catalog/compilations/{cid}").json()
    assert comp_after == comp_before
    assert client.get(f"/v1/runs/{data['run_id']}").json() == run_before
    assert client.get(f"/v1/events/{event['event_id']}").json() == event_before


# --------------------------------------------------------------------------- #
# calibrated master run: prediction applies the run's P calibration offset
# --------------------------------------------------------------------------- #
def test_prediction_applies_master_run_calibration_terms(client):
    from tests.test_calibration import (  # noqa: E402
        P_BIAS,
        S_BIAS,
        _cal_request,
        _event_window as _cal_event_window,
        _post_events,
    )

    _post_events(client, range(0, 5), "CA", P_BIAS, S_BIAS)
    _post_events(client, [7], "CA", P_BIAS, S_BIAS)
    created = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="cal-gap"),
    )
    assert created.status_code == 201, created.text
    activated = client.post(
        "/v1/models/default/1.0/calibrations/cal-gap/activate"
    )
    assert activated.status_code == 200

    window = _cal_event_window(11, "CA", P_BIAS, S_BIAS)
    data = client.post(
        "/v1/process", json={**window, "calibration": {"version": "cal-gap"}}
    ).json()
    assert data["calibration"] == {"version": "cal-gap", "status": "active"}
    event = _best(data)
    comp = _publish_compilation(
        client,
        start=datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=10),
        days=5,
    )
    rep = _assess(client, comp["compilation_id"], tolerance=5.0).json()

    frozen_entry = rep["frozen"]["entries"][0]
    assert frozen_entry["calibration"]["version"] == "cal-gap"
    p_terms = {t["station"]: t["offset_s"]
               for t in frozen_entry["calibration"]["p_terms"]}
    assert p_terms  # P offsets were frozen with the report

    events = client.get(
        f"/v1/pick-gap-assessments/{rep['assessment_id']}/events"
    ).json()["events"]
    assert events, "calibrated event produced evidence rows"
    assert rep["hit_count"] >= 1
    origin_t = to_epoch(
        datetime.fromisoformat(event["origin_time"].replace("Z", "+00:00"))
        .astimezone(timezone.utc).replace(tzinfo=None)
    )
    for row in events[:3]:
        # predicted == origin + travel time + frozen P offset (rounded)
        expect_pred = origin_t + row["travel_time_p_s"] + p_terms[row["station"]]
        got = (
            datetime.strptime(row["predicted_p_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
            - datetime(1970, 1, 1)
        ).total_seconds()
        assert abs(got - expect_pred) < 2e-3
        assert row["calibration"]["version"] == "cal-gap"
        assert row["p_calibration_offset_s"] == round(p_terms[row["station"]], 6)
