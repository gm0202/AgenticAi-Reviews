# Agentic AI Review Trend Analyzer

## 🚀 Overview
The **Agentic AI Review Trend Analyzer** is an intelligent system designed to scrape Google Play Store reviews, identify evolving topics (issues, feature requests, feedback), and generate daily trend analysis reports.

Unlike traditional keyword-based approaches, this system uses **Agentic AI (Groq/Llama 3)** to semantically understand user intent and consolidate similar topics (e.g., merging "Rude driver" and "Impolite delivery partner" into a single category).

## ✨ Features
*   **Automated Scraper**: Fetches daily batches of reviews from the Google Play Store.
*   **Agentic Topic Extraction**: Uses LLM (Llama 3.3-70b) to extract precise issues from raw text.
*   **Dynamic Taxonomy**: Auto-learns and adapts to new topics over time.
*   **Semantic Deduplication**: Uses vector embeddings to merge semantically similar topics, ensuring high recall.
*   **Trend Reporting**: Generates sliding-window trend matrices (Topic vs. Date).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gm0202/AgenticAi-Reviews.git
    cd PulseGen
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API Key:
    ```env
    GROQ_API_KEY=gsk_...
    ```

## 🏃 Usage

### 1. Data Ingestion
Fetch reviews for a specific date (defaults to widely recent data for testing):
```bash
python src/scraper.py
```
*Data is saved to `data/YYYY-MM-DD.json`.*

### 2. Run the Agent
Process the daily reviews to extract and map topics:
```bash
python src/agent.py
```
*   Extracts topics using the LLM.
*   Updates `taxonomy.json` with new findings.
*   Saves daily statistics to `output/stats_YYYY-MM-DD.json`.

### 3. Generate Report
Create the final Trend Analysis Report:
```bash
python src/analyzer.py
```
*   Generates `output/trend_report.csv` and `output/trend_report.md`.

## 📂 Project Structure
```
PulseGen/
├── data/                 # Raw daily review JSONs
├── output/               # Processed stats and final reports
├── src/
│   ├── scraper.py        # Google Play Store Scraper
│   ├── agent.py          # Core Agent (LLM + Embeddings)
│   └── analyzer.py       # Report Generator
├── taxonomy.json         # Persistent topic memory (Vector Store)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🔮 Future Roadmap
*   **Backend**: Deploy as a FastAPI service on Railway with Celery workers for async processing.
*   **Frontend**: Next.js dashboard on Vercel for interactive data visualization (Heatmaps/Charts).
*   **Database**: Migrate local JSON storage to PostgreSQL with pgvector.
