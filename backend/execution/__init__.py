"""Order execution adapters."""

from .base import ExecutionAdapter, Fill
from .paper import PaperExecution

__all__ = ["ExecutionAdapter", "Fill", "PaperExecution", "create_execution"]


def create_execution(settings, bus):  # type: ignore[no-untyped-def]
    """Build the execution adapter the configuration actually permits.

    Live execution requires passing the three-part gate in
    :meth:`backend.config.Settings.live_gate_status`. Anything short of that
    returns the paper engine — the system never silently upgrades to real money.
    """
    from .live_ccxt import LiveExecution, LiveTradingRefused

    allowed, reason = settings.live_gate_status()
    if not allowed:
        return PaperExecution(settings, bus), reason

    try:
        return LiveExecution(settings, bus), reason
    except LiveTradingRefused as exc:
        bus.emit(
            "system.live_refused",
            f"Live trading refused: {exc}. Falling back to paper execution.",
            severity="warning",
        )
        return PaperExecution(settings, bus), str(exc)
