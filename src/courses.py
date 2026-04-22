"""Canonical race course definitions — Session 19.

Extracted from px4_v51_baseline.py so tools can import courses without
dragging in mavsdk. The numbers here match px4_v51_baseline.COURSES
verbatim. If px4_v51_baseline is updated, update both (or flip that
module to import from here — deferred cleanup).

Each course is a list of (N, E, D) tuples in NED meters. `D` is
negative for altitude (NED convention: Down is positive).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

GateNED = Tuple[float, float, float]

COURSES: Dict[str, List[GateNED]] = {
    "sprint": [
        (30, 0, -3), (50, 15, -3), (80, 15, -3), (100, 0, -3), (130, 0, -3),
        (130, 20, -3), (100, 20, -3), (70, 30, -3), (30, 30, -3), (0, 15, -3),
    ],
    "technical": [
        (8, 0, -2.5), (12, 6, -2.5), (8, 12, -2.5), (0, 12, -2.5), (-4, 6, -2.5),
        (0, 0, -2.5), (6, -4, -2.5), (14, -4, -2.5), (18, 0, -2.5), (14, 4, -2.5),
        (8, 4, -2.5), (4, 0, -3),
    ],
    "mixed": [
        (20, 0, -3), (35, 10, -3), (50, 10, -3), (55, 20, -3), (40, 28, -3),
        (20, 28, -3), (8, 20, -3), (8, 10, -3), (15, 2, -3), (25, -2, -3),
        (35, 2, -3), (20, 5, -3),
    ],
}


def list_courses() -> List[str]:
    return sorted(COURSES.keys())


def get_course(name: str) -> List[GateNED]:
    """Return a course by name. Raises KeyError with a helpful message."""
    if name not in COURSES:
        raise KeyError(
            f"Unknown course {name!r}. Known: {list_courses()}"
        )
    return list(COURSES[name])
