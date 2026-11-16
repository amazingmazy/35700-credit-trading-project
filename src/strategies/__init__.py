"""Strategy package. Import new strategy modules here so @register runs."""
from src.strategies.base import STRATEGY_REGISTRY, Strategy, WeightStrategy, register
from src.strategies import baselines  # noqa: F401  (registers baseline strategies)
from src.strategies.baselines import BuyAndHold, EqualWeightLong, MomentumLongShort, NoTrade

__all__ = [
    "STRATEGY_REGISTRY", "Strategy", "WeightStrategy", "register",
    "NoTrade", "BuyAndHold", "EqualWeightLong", "MomentumLongShort",
]
