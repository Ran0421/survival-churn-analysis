from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt

def fit_aft_model(df: pd.DataFrame):
    """
    AFT model — alternative to Cox
    Instead of hazard ratios, gives you TIME ratios
    'Feature X multiplies expected survival time by Y'
    More interpretable for business stakeholders
    """
    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(df, duration_col='duration', event_col='event')
    
    print("=== Weibull AFT Model Summary ===")
    aft.print_summary()
    
    c_index = aft.concordance_index_
    print(f"\nAFT C-index: {c_index:.4f}")
    return aft

def compare_cox_vs_aft(cox_c, aft_c):
    print("\n=== Model Comparison ===")
    print(f"Cox PH C-index:      {cox_c:.4f}")
    print(f"Weibull AFT C-index: {aft_c:.4f}")
    winner = "Cox PH" if cox_c > aft_c else "Weibull AFT"
    print(f"Better model: {winner}")