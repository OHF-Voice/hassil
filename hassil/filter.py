"""Filter an :class:`~hassil.intents.Intents` object for a specific Home Assistant.

The intents shipped for a language describe *every* possible command, but a
given Home Assistant instance only has a subset of entity domains, intents,
etc. Narrowing the intents to what a home actually supports keeps recognition
(and especially fuzzy-model training) focused on relevant commands.
"""

from dataclasses import dataclass, field, replace
from typing import Collection, Dict, List, Optional, Set

from .intents import Intent, IntentData, Intents


@dataclass
class HomeInfo:
    """Information about a user's Home Assistant instance used for filtering."""

    domains: Optional[Set[str]] = None
    """Entity domains that are available (e.g., ``{"light", "switch"}``).

    ``None`` means "unknown" and disables domain filtering.
    """

    intent_names: Optional[Set[str]] = None
    """Intents that are supported (e.g., ``{"HassTurnOn", "HassTurnOff"}``).

    ``None`` means "unknown" and disables intent filtering.
    """

    extra_domains: Set[str] = field(default_factory=set)
    """Domains to always treat as available, in addition to :attr:`domains`."""

    def available_domains(self) -> Optional[Set[str]]:
        """Effective set of available domains, or ``None`` if unknown."""
        if self.domains is None:
            if self.extra_domains:
                return set(self.extra_domains)
            return None

        return self.domains | self.extra_domains


def filter_intents(
    intents: Intents,
    *,
    domains: Optional[Collection[str]] = None,
    intent_names: Optional[Collection[str]] = None,
    home: Optional[HomeInfo] = None,
    keep_domainless: bool = True,
) -> Intents:
    """Return a copy of ``intents`` scoped to a Home Assistant instance.

    A block of sentences (:class:`~hassil.intents.IntentData`) is kept when:

    - its intent is in ``intent_names`` (if provided), and
    - it is not scoped to a domain, or at least one of its
      :attr:`~hassil.intents.IntentData.domains` is available.

    Args:
        intents: Intents to filter (not modified).
        domains: Available entity domains. ``None`` disables domain filtering.
        intent_names: Supported intent names. ``None`` keeps all intents.
        home: Convenience source for ``domains``/``intent_names``. Explicit
            ``domains``/``intent_names`` arguments take precedence.
        keep_domainless: Keep blocks that are not scoped to any domain (e.g.,
            name-only commands whose domain is resolved at runtime). Only has
            an effect when domain filtering is active.

    Returns:
        A new :class:`~hassil.intents.Intents` sharing slot lists and expansion
        rules with the original, containing only the matching intents/blocks.
    """
    if home is not None:
        if domains is None:
            available_domains = home.available_domains()
        else:
            available_domains = set(domains)

        if intent_names is None and home.intent_names is not None:
            intent_names = home.intent_names
    else:
        available_domains = set(domains) if domains is not None else None

    available_domains = (
        set(available_domains) if available_domains is not None else None
    )
    allowed_intents = set(intent_names) if intent_names is not None else None

    filtered_intents: Dict[str, Intent] = {}
    for intent_name, intent in intents.intents.items():
        if (allowed_intents is not None) and (intent_name not in allowed_intents):
            continue

        kept_data: List[IntentData] = [
            data
            for data in intent.data
            if _keep_data(data, available_domains, keep_domainless)
        ]

        if kept_data:
            filtered_intents[intent_name] = Intent(name=intent_name, data=kept_data)

    return replace(intents, intents=filtered_intents)


def _keep_data(
    data: IntentData,
    available_domains: Optional[Set[str]],
    keep_domainless: bool,
) -> bool:
    """Return True if a block of sentences should be kept."""
    if available_domains is None:
        # Domain filtering disabled
        return True

    block_domains = data.domains
    if not block_domains:
        # Not scoped to any domain
        return keep_domainless

    # Keep if any of the block's domains is available
    return bool(block_domains & available_domains)
