import os
import json
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "output"
REPORT_FILE_CSV = os.path.join(OUTPUT_DIR, "trend_report.csv")
REPORT_FILE_MD = os.path.join(OUTPUT_DIR, "trend_report.md")

def generate_report():
    print("[ANALYZER] Generating Trend Report...")
    
    # 1. Gather all daily stats
    all_data = [] # List of dicts: {'date': '...', 'topic': '...', 'count': N}
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"[ERROR] Output directory '{OUTPUT_DIR}' not found.")
        return

    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("stats_") and f.endswith(".json")]
    
    if not files:
        print("[ANALYZER] No stats files found to analyze.")
        return

    for f in files:
        # Filename format: stats_YYYY-MM-DD.json
        date_str = f.replace("stats_", "").replace(".json", "")
        file_path = os.path.join(OUTPUT_DIR, f)
        
        with open(file_path, "r") as json_file:
            daily_stats = json.load(json_file)
            
        for topic, count in daily_stats.items():
            all_data.append({
                "Date": date_str,
                "Topic": topic,
                "Count": count
            })

    # 2. Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("[ANALYZER] No data found in stats files.")
        return

    # 3. Pivot: Rows=Topic, Cols=Date
    pivot_df = df.pivot_table(index="Topic", columns="Date", values="Count", fill_value=0)
    
    # Sort columns by date (though pivot usually handles this if dates are ISO strings)
    pivot_df = pivot_df.sort_index(axis=1)
    
    # Add a "Total" column for sorting top topics
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by="Total", ascending=False)
    
    # Drop Total for the final time-series view (optional, but keep it for ranking)
    # Let's keep it for context, or drop it if strictly T-> T-30
    
    print("\n--- TOP 10 TRENDING TOPICS ---")
    print(pivot_df.head(10)[["Total"]])

    # 4. Export
    # CSV
    pivot_df.to_csv(REPORT_FILE_CSV)
    print(f"[SAVED] CSV Report: {REPORT_FILE_CSV}")
    
    # Markdown
    with open(REPORT_FILE_MD, "w", encoding="utf-8") as f:
        f.write(f"# Trend Analysis Report (Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
        f.write(pivot_df.to_markdown())
        
    print(f"[SAVED] Markdown Report: {REPORT_FILE_MD}")

if __name__ == "__main__":
    generate_report()
