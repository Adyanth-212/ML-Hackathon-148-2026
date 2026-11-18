"""Smoke-test every trip exposed by the running PathSnap API.

    python scripts/smoke_test_api.py                 # evaluator trips (the demo path), default
    python scripts/smoke_test_api.py student-test    # practice set

PATHSNAP_API_URL points at Laptop 1 (default http://127.0.0.1:8000). Only pass/fail and timing
are printed, not scores, so nobody looks at evaluator results before the demo.
"""
from __future__ import annotations

import os
import sys
import time

import requests


API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.environ.get("PATHSNAP_SMOKE_TIMEOUT", "300"))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "evaluator"


def main() -> int:
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        health.raise_for_status()
        trips_response = requests.get(f"{API_URL}/{PREFIX}/trips", timeout=30)
        trips_response.raise_for_status()
        trips = trips_response.json()
    except requests.RequestException as exc:
        print(f"API connection failed: {exc}", file=sys.stderr)
        return 1

    if not trips:
        print(f"No trips were returned by /{PREFIX}/trips", file=sys.stderr)
        return 1

    print(f"PathSnap API v{health.json().get('version')} · {len(trips)} trips")
    failures = []
    for trip in trips:
        trip_id = trip["trip_id"]
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{API_URL}/{PREFIX}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT
            )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            result = response.json()
            if len(result.get("pred_arcs", [])) != trip["n_observations"]:
                raise AssertionError("pred_arcs length does not match GPS observation count")
            if not result.get("gps"):
                raise AssertionError("response contains no GPS coordinates")
            if "metrics" not in result:
                raise AssertionError("response contains no server-side metrics")
            print(f"PASS {trip_id}: {trip['n_observations']} points in {elapsed:.2f}s")
        except (requests.RequestException, ValueError, AssertionError) as exc:
            failures.append((trip_id, str(exc)))
            print(f"FAIL {trip_id}: {exc}")

    print(f"\nResult: {len(trips) - len(failures)}/{len(trips)} passed")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
