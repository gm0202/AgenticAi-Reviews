import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_stats_data(output_dir):
    files = glob(os.path.join(output_dir, "stats_*.json"))
    data = {}
    for f in files:
        # filename is expected to be stats_YYYY-MM-DD.json
        basename = os.path.basename(f)
        date_str = basename.replace("stats_", "").replace(".json", "")
        with open(f, "r") as file:
            data[date_str] = json.load(file)
    
    # Convert to DataFrame: rows=Topics, cols=Dates
    df = pd.DataFrame(data).fillna(0)
    return df

def plot_top_topics_bar(df, output_dir, date_col=None):
    if df.empty: return
    
    # If specific date provided, plot bar chart for that date
    if date_col and date_col in df.columns:
        series = df[date_col].sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=series.values, y=series.index, hue=series.index, legend=False, palette="viridis")
        plt.title(f"Top 20 Topics for {date_col}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_topics_{date_col}.png"))
        plt.close()

def plot_heatmap(df, output_dir):
    if df.empty: return
    
    # Filter to top 30 topics by total count across all days
    df_sum = df.sum(axis=1).sort_values(ascending=False).head(30)
    df_top = df.loc[df_sum.index]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_top, annot=True, fmt="g", cmap="YlGnBu")
    plt.title("Topic Frequency Heatmap (Top 30)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topics_heatmap.png"))
    plt.close()

def generate_visualizations(output_dir, current_date=None):
    """
    Generates visualizations based on the JSON stats files in the output directory.
    
    Args:
        output_dir (str): Path to the directory containing stats_*.json files.
        current_date (str, optional): The specific date to generate a focused bar chart for.
    """
    try:
        df = load_stats_data(output_dir)
        if df.empty:
            print("[VISUALIZATION] No data found to visualize.")
            return

        print(f"[VISUALIZATION] Generating plots for {df.shape[1]} days and {df.shape[0]} topics...")
        
        # 1. Heatmap of top topics over time
        plot_heatmap(df, output_dir)
        
        # 2. Bar chart for the specific date (or latest)
        if current_date:
            plot_top_topics_bar(df, output_dir, current_date)
        else:
            latest_date = df.columns.max()
            plot_top_topics_bar(df, output_dir, latest_date)

        print("[VISUALIZATION] Plots saved to output directory.")
        
    except Exception as e:
        print(f"[VISUALIZATION] Error generating plots: {e}")
