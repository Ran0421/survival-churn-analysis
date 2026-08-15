import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_overall_km(durations, events, save_path='outputs/km_overall.png'):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label='All Customers')
    
    ax = kmf.plot_survival_function(ci_show=True, figsize=(10, 6))
    ax.set_title('Kaplan-Meier Survival Curve — Customer Churn', fontsize=14)
    ax.set_xlabel('Tenure (Months)')
    ax.set_ylabel('Survival Probability S(t)')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Median survival')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    median_survival = kmf.median_survival_time_
    if median_survival == float('inf'):
      print("Median survival time: >72 months (fewer than 50% of customers churned)")
      print("→ The majority of customers remain active throughout the observation window")
    else:
      print(f"Median survival time: {median_survival:.1f} months")
      print(f"→ 50% of customers churn within {median_survival:.1f} months")
    return kmf

def plot_km_by_group(df, durations, events, group_col, save_path=None):
    """Compare survival curves between groups — most useful for interviews"""
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = df[group_col].unique()
    group_durations = {}
    group_events    = {}
    
    for group in sorted(groups):
        mask = df[group_col] == group
        d = durations[mask]
        e = events[mask]
        group_durations[group] = d
        group_events[group]    = e
        kmf.fit(d, event_observed=e, label=f'{group_col}={group}')
        kmf.plot_survival_function(ax=ax, ci_show=True)
    
    # Log-rank test — tells you if groups are statistically different
    if len(groups) == 2:
        g = sorted(groups)
        result = logrank_test(
            group_durations[g[0]], group_durations[g[1]],
            event_observed_A=group_events[g[0]],
            event_observed_B=group_events[g[1]]
        )
        ax.set_title(
            f'KM Curves by {group_col}\n'
            f'Log-rank test p-value: {result.p_value:.4f} '
            f'({"significant" if result.p_value < 0.05 else "not significant"})',
            fontsize=13
        )
    else:
        ax.set_title(f'KM Curves by {group_col}', fontsize=13)
    
    ax.set_xlabel('Tenure (Months)')
    ax.set_ylabel('Survival Probability S(t)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()