"""Independent, reproducible channel-fading processes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random


EndpointKey = tuple[str, int]
ChannelLinkKey = tuple[str, EndpointKey, EndpointKey]


@dataclass(frozen=True)
class ChannelFadingConfig:
    enabled: bool = False
    model: str = "correlated_lognormal"
    seed: int = 2026
    sigma_db: float = 2.0
    rho: float = 0.95


def make_channel_link_key(
    link_type: str,
    src: EndpointKey,
    dst: EndpointKey,
    *,
    undirected: bool,
) -> ChannelLinkKey:
    """Return a stable physical-link key without relying on Python hashing."""

    if undirected and dst < src:
        src, dst = dst, src
    return str(link_type), src, dst


class CorrelatedLognormalFading:
    """Per-link AR(1) shadow fading in dB with isolated RNG streams."""

    def __init__(self, config: ChannelFadingConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = int(config.seed if seed is None else seed)
        self._states_db: dict[ChannelLinkKey, float] = {}
        self._last_slots: dict[ChannelLinkKey, int] = {}
        self._generators: dict[ChannelLinkKey, random.Random] = {}

    def _generator_for(self, link_key: ChannelLinkKey) -> random.Random:
        generator = self._generators.get(link_key)
        if generator is not None:
            return generator
        payload = repr((self.seed, link_key)).encode("utf-8")
        derived_seed = int.from_bytes(
            hashlib.blake2b(payload, digest_size=16).digest(), "big"
        )
        generator = random.Random(derived_seed)
        self._generators[link_key] = generator
        return generator

    def loss_db(self, link_key: ChannelLinkKey, time_slot: int) -> float:
        """Return this link's fading loss at an absolute simulation slot."""

        if not self.config.enabled:
            return 0.0
        if self.config.model != "correlated_lognormal":
            raise ValueError(f"Unsupported channel fading model {self.config.model!r}.")

        slot = int(time_slot)
        if slot < 0:
            raise ValueError("Channel fading time_slot must be non-negative.")
        generator = self._generator_for(link_key)
        sigma_db = float(self.config.sigma_db)
        rho = float(self.config.rho)
        innovation_scale = math.sqrt(max(1.0 - rho * rho, 0.0))

        if link_key not in self._states_db:
            state_db = generator.gauss(0.0, sigma_db)
            last_slot = 0
        else:
            state_db = self._states_db[link_key]
            last_slot = self._last_slots[link_key]
            if slot < last_slot:
                raise ValueError(
                    f"Channel fading time moved backwards for {link_key!r}: "
                    f"{slot} < {last_slot}."
                )

        for _ in range(last_slot, slot):
            epsilon_db = generator.gauss(0.0, sigma_db)
            state_db = rho * state_db + innovation_scale * epsilon_db

        if not math.isfinite(state_db):
            raise FloatingPointError(f"Non-finite fading state for {link_key!r}.")
        self._states_db[link_key] = state_db
        self._last_slots[link_key] = slot
        return state_db


__all__ = [
    "ChannelFadingConfig",
    "ChannelLinkKey",
    "CorrelatedLognormalFading",
    "EndpointKey",
    "make_channel_link_key",
]
