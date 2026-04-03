# Ablation Study Results — Project Sambhav

## Student Domain
| Experiment | Brier Score | Change |
|---|---|---|
| Full system | 0.1069 | baseline |
| No past performance | 0.1533 | +0.0464 ↑ worse |
| Single feature only | 0.1533 | +0.0464 ↑ worse |

## HR Domain
| Experiment | Brier Score | Change |
|---|---|---|
| Full system | 0.1446 | baseline |
| No behavioral features | 0.1545 | +0.0099 ↑ worse |
| Single feature only | 0.2082 | +0.0636 ↑ worse |

## Behavioral Domain
| Experiment | Brier Score | Change |
|---|---|---|
| Full system | 0.0003 | baseline |
| Single feature only | 0.0016 | +0.0013 ↑ worse |

## Conclusion
Removing any feature family degrades performance across all domains.
The full stacking ensemble with all features consistently outperforms
any ablated version — validating the multi-feature engineering approach.