"""Sim backends for the AI Grand Prix race stack.

Public surface:

    from sim import (
        SimAdapter, SimState, SimCapability, SimInfo, IMUReading,
        PX4SITLAdapter, DCLSimAdapter, MockKinematicAdapter, make_adapter,
    )

See adapter.py for the design notes.
"""

from .adapter import (
    SimAdapter,
    SimState,
    SimCapability,
    SimInfo,
    IMUReading,
    PX4SITLAdapter,
    DCLSimAdapter,
    make_adapter,
)
from .mock import MockKinematicAdapter

__all__ = [
    "SimAdapter",
    "SimState",
    "SimCapability",
    "SimInfo",
    "IMUReading",
    "PX4SITLAdapter",
    "DCLSimAdapter",
    "MockKinematicAdapter",
    "make_adapter",
]
