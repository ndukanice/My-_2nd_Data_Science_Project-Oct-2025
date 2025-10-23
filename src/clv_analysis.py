from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

def load_full() -> pd.DataFrame:
    """Load the full processed dataset with CLV"""
    return pd.read_csv(PROC_DIR / "processed_full.csv")

def add_clv_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add CLV quartile labels to dataframe"""
    df = df.copy()
    df["CLV_quartile"] = pd.qcut(df["CLV"], q=4, labels=["Low", "Med", "High", "Premium"])
    return df

def churn_by_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate churn rate by CLV quartile"""
    tmp = add_clv_quartiles(df)
    out = tmp.groupby("CLV_quartile")["Churn"].mean().reset_index()
    out["ChurnRate"] = (out["Churn"] * 100).round(2)
    return out[["CLV_quartile", "ChurnRate"]]

def generate_clv_report():
    """Generate CLV analysis report with visualizations"""
    try:
        df = load_full()
    except FileNotFoundError:
        print("ERROR: processed_full.csv not found.")
        print("Run this first: python src\\data_prep.py")
        return
    
    df_with_quartiles = add_clv_quartiles(df)
    churn_rates = churn_by_quartile(df)
    
    print("=" * 60)
    print("CLV ANALYSIS REPORT")
    print("=" * 60)
    print("\nChurn Rate by CLV Quartile:")
    print(churn_rates.to_string(index=False))
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CLV Distribution
    axes[0].hist(df["CLV"], bins=30, color="skyblue", edgecolor="black")
    axes[0].set_title("CLV Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("CLV ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Plot 2: Churn Rate by CLV Quartile
    sns.barplot(
        data=churn_rates, 
        x="CLV_quartile", 
        y="ChurnRate", 
        palette="viridis",
        ax=axes[1]
    )
    axes[1].set_title("Churn Rate by CLV Quartile", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("CLV Quartile")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = ROOT / "clv_analysis_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved: {output_path}")
    
    # Business Insights
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)
    
    low_churn = churn_rates[churn_rates["CLV_quartile"] == "Low"]["ChurnRate"].values[0]
    premium_churn = churn_rates[churn_rates["CLV_quartile"] == "Premium"]["ChurnRate"].values[0]
    
    print(f"• Low CLV customers churn at {low_churn}% rate")
    print(f"• Premium CLV customers churn at {premium_churn}% rate")
    print("• Prioritize retention for high-CLV, high-risk segments")
    print("• Contract type, Tenure, and MonthlyCharges are key drivers")
    print("• Target month-to-month, fiber-optic users lacking support services")
    print("=" * 60)

if __name__ == "__main__":
    generate_clv_report()