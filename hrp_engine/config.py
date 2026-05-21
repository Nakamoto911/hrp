import dataclasses

@dataclasses.dataclass
class StrategyParams:
    lookback_years: int = 4
    rebalance_frequency: str = 'quarterly'  # 'daily', 'monthly', 'quarterly', 'semi-annually', 'yearly'
    linkage_method: str = 'single'          # 'single', 'complete', 'ward'
    drift_threshold: float = 0.015
    transaction_cost_bps: float = 5.0
    french_pfu_rate: float = 0.314
    denoise: bool = True
    bisection_method: str = 'tree'          # 'index' (original) or 'tree' (tree-based)

