"""Bridging a loaded policy into the trading loop.

Deliberately small, and deliberately not wired into ``BotAgent`` yet.

The obstacle is a real one rather than an oversight. ``BotAgent`` hands a
strategy a list of candles; a learned policy needs a *feature vector*, and the
feature builder currently lives in ``research/`` — which ``backend`` must never
import, since the whole point of the one-way rule is that the trading server
cannot be broken by research dependencies. Wiring a learned policy into the live
loop therefore requires moving the feature registry into ``backend`` with a
parity test proving the two produce identical vectors. That is deployment work,
and it belongs with Phase 8 rather than being half-done here.

What this class does today is define the seam: give it a bundle and something
that can produce the observation vector, and it answers with a
:class:`SignalAction`. Offline evaluation goes through ``research`` instead,
where the environment already builds observations.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ..models import SignalAction
from .loader import PolicyBundle

logger = logging.getLogger("jojo.policy.adapter")

#: Must match ``research.policy.PolicyAction``. The integer codes are stored in
#: the database and baked into every exported network, so they are frozen.
ACTION_ORDER: tuple[SignalAction, ...] = (
    SignalAction.HOLD,
    SignalAction.LONG,
    SignalAction.SHORT,
    SignalAction.CLOSE,
)


class PolicyRunner:
    """Turns an observation into a :class:`SignalAction`, with the contract checked."""

    def __init__(
        self,
        bundle: PolicyBundle,
        observation_builder: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self.bundle = bundle
        self.observation_builder = observation_builder
        if bundle.policy.weights.action_size != len(ACTION_ORDER):
            raise ValueError(
                f"{bundle.policy_id} has {bundle.policy.weights.action_size} actions but the "
                f"runtime knows {len(ACTION_ORDER)}: {[a.value for a in ACTION_ORDER]}"
            )

    @property
    def name(self) -> str:
        return self.bundle.policy_id

    def decide(self, observation: np.ndarray) -> SignalAction:
        vector = np.asarray(observation, dtype=np.float64)
        if vector.shape != (self.bundle.observation_size,):
            raise ValueError(
                f"{self.bundle.policy_id} expects {self.bundle.observation_size} inputs, "
                f"got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            # Refusing beats guessing: a NaN reaching the network produces a
            # confident-looking action derived from nothing.
            logger.warning("%s: non-finite observation, holding", self.bundle.policy_id)
            return SignalAction.HOLD
        return ACTION_ORDER[self.bundle.act(vector)]

    def probabilities(self, observation: np.ndarray) -> dict[str, float]:
        values = self.bundle.policy.probabilities(np.asarray(observation, dtype=np.float64))
        return {action.value: float(p) for action, p in zip(ACTION_ORDER, values)}
