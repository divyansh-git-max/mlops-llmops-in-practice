import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path="data/churn.csv", output_dir="output"):
    # Create output directory automatically if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving visualizations to {output_dir}/")

    # Load data
    print(f"📊 Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_path}. Make sure you run this script from the dataVersioning folder.")
        return

    # Set visualization style
    sns.set_theme(style="whitegrid")

    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Target Variable Distribution: Churn')
    plt.savefig(f'{output_dir}/01_churn_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved 01_churn_distribution.png")

    # 2. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set2', bins=30)
    plt.title('Tenure Distribution by Churn')
    plt.xlabel('Tenure (Months)')
    plt.savefig(f'{output_dir}/02_tenure_vs_churn.png', bbox_inches='tight')
    plt.close()
    print("Saved 02_tenure_vs_churn.png")

    # 3. Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='Set2', common_norm=False)
    plt.title('Monthly Charges Distribution by Churn')
    plt.savefig(f'{output_dir}/03_monthly_charges_vs_churn.png', bbox_inches='tight')
    plt.close()
    print("Saved 03_monthly_charges_vs_churn.png")

    # 4. Contract Type vs Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Churn by Contract Type')
    plt.savefig(f'{output_dir}/04_contract_vs_churn.png', bbox_inches='tight')
    plt.close()
    print("Saved 04_contract_vs_churn.png")

    # 5. Correlation Matrix
    # Telco Churn 'TotalCharges' is a string by default, we must convert it to numeric for correlation
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig(f'{output_dir}/05_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Saved 05_correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()
