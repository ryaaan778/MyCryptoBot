"""Model-driven trading: the agents decide, the risk engine still governs.

This package is what turns the five bots from strategy executors into agents
that reason about their market, read the news, remember what they tried, and
change their minds. The pieces:

- :mod:`.schema`    what the model is allowed to say — direction and conviction,
                    never size
- :mod:`.client`    the provider connection, with server-side web search
- :mod:`.prompts`   desk state rendered as context
- :mod:`.memory`    an append-only record of decisions and their outcomes, fed
                    back so the agent learns from its own history
- :mod:`.brain`     cadence, budget and the current stance
- :mod:`.strategy`  the ``Strategy`` adapter, so nothing downstream changes

**The boundary that does not move.** A stance goes through exactly the same path
as a hand-written strategy's signal: ``orchestrator.handle_signal`` →
``risk.size_position`` → ``risk.can_open`` → order. The model has no way to
express a size, no way to reach the risk engine, and no way to place an order.
It argues for a direction; the desk decides what that is worth.

Turning it on is one flag (``llm.enabled``) plus an API key. Leaving it off
costs nothing and changes nothing — nothing in this package is imported unless a
bot's strategy is ``"llm"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .brain import AccountContext, LLMBrain
from .client import (
    DEFAULT_MODEL,
    AnthropicDecisionClient,
    DecisionClient,
    LLMUnavailable,
    ScriptedDecisionClient,
    Usage,
)
from .memory import DecisionRecord, LLMMemory, memory_path
from .schema import TradeStance
from .strategy import LLMStrategy

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..models import BotConfig

logger = logging.getLogger("jojo.llm")

__all__ = [
    "AccountContext", "AnthropicDecisionClient", "DEFAULT_MODEL",
    "DecisionClient", "DecisionRecord", "LLMBrain", "LLMMemory",
    "LLMStrategy", "LLMUnavailable", "ScriptedDecisionClient", "TradeStance",
    "Usage", "build_client", "build_brain", "memory_path",
]


def build_client(settings: "Settings", *, client: object | None = None) -> DecisionClient:
    """Construct the provider client named in settings."""
    llm = settings.llm
    provider = (llm.provider or "anthropic").strip().lower()
    if provider != "anthropic":
        raise ValueError(
            f"unknown LLM provider {llm.provider!r}; only 'anthropic' is wired up"
        )
    return AnthropicDecisionClient(
        model=llm.model, api_key=llm.api_key or None,
        max_tokens=llm.max_tokens, effort=llm.effort,
        web_search=llm.web_search, max_searches=llm.max_searches,
        timeout_sec=llm.timeout_sec, client=client,
    )


def _persona_line(config: "BotConfig") -> str:
    """The character's own framing, so five agents don't converge on one voice.

    This is a *disposition*, not a rule: a contrarian is still free to go with
    the trend when the evidence says so. Making it a hard constraint would just
    be five hard-coded strategies again, which is the thing we left behind.
    """
    persona = config.persona
    bits = [b for b in (persona.title, persona.stand) if b]
    if not bits:
        return ""
    line = " · ".join(bits)
    if persona.quote:
        line += f'\nYour outlook: "{persona.quote}"'
    return (
        f"{line}\nThat is your disposition, not a rule. It shapes what you look "
        "at first, never what you are allowed to conclude."
    )


def build_brain(
    config: "BotConfig", settings: "Settings", *, client: DecisionClient | None = None
) -> LLMBrain:
    """Wire one bot's brain from configuration."""
    llm = settings.llm
    return LLMBrain(
        bot_id=config.id, bot_name=config.name,
        client=client or build_client(settings),
        data_dir=settings.data_dir, persona=_persona_line(config),
        decide_every_sec=llm.decide_every_sec, stance_ttl_sec=llm.stance_ttl_sec,
        min_confidence=llm.min_confidence, daily_call_budget=llm.daily_call_budget,
        memory_window=llm.memory_window, web_search=llm.web_search,
    )
