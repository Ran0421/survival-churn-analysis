import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

def fit_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """
    df must contain 'duration' and 'event' columns
    plus all feature columns
    """
    cph = CoxPHFitter(penalizer=0.1)  # L2 regularization
    cph.fit(df, duration_col='duration', event_col='event')
    
    print("=== Cox PH Model Summary ===")
    cph.print_summary()
    return cph

def plot_hazard_ratios(cph: CoxPHFitter, save_path='outputs/hazard_ratios.png'):
    """
    Hazard ratios — your SHAP equivalent for survival analysis
    HR > 1 → increases churn risk
    HR < 1 → decreases churn risk  
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cph.plot(ax=ax)
    ax.set_title('Cox PH Model — Hazard Ratios per Feature\n'
                 '(HR > 1 = higher churn risk, HR < 1 = lower churn risk)',
                 fontsize=13)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def predict_survival_for_segments(cph, X, durations, events):
    """
    Predict survival probability at specific time points
    for different customer segments
    """
    time_points = [6, 12, 24, 36, 48]  # months
    
    results = {}
    for t in time_points:
        survival_probs = cph.predict_survival_function(X, times=[t])
        results[f'P(survive>{t}mo)'] = survival_probs.mean(axis=1).values[0]
    
    print("\n=== Predicted Survival Probabilities ===")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    
    return results

def check_proportional_hazards(cph, df):
    """
    Validate Cox model assumption — proportional hazards
    p < 0.05 means assumption is violated for that feature
    """
    result = proportional_hazard_test(cph, df, time_transform='rank')
    print("\n=== Proportional Hazards Assumption Test ===")
    print(result.summary)
    return result

def get_c_index(cph) -> float:
    """
    Concordance index — main evaluation metric
    0.5 = random, 1.0 = perfect, >0.7 = good
    Equivalent to AUC for survival models
    """
    c_index = cph.concordance_index_
    print(f"\nC-index (Concordance Index): {c_index:.4f}")
    print(f"Interpretation: model correctly orders "
          f"{c_index*100:.1f}% of customer pairs by churn risk")
    return c_index