# SEC Airline RAG Pipeline & Telemetry Analytics

A local data science pipeline that programmatically extracts, prunes, and processes unstructured narrative financial data from large, 100+ page SEC 10-K filings (FY2024 for United Airlines and Delta Air Lines) to evaluate Natural Language Processing (NLP) inference efficiency.

## Tech Stack & Architecture
- **Language:** Python 3.12 (Pandas, PyPDF)
- **Analytics & BI:** Tableau Desktop Public Edition
- **Version Control:** Git & GitHub

## Key Data Science Core Metrics Logged
1. **Inference Latency:** Precise compute execution time per query category.
2. **Text Processing Volumes:** Character-to-token distribution maps capturing raw input string length.
3. **Infrastructure Cost Modeling:** Simulating operational localized hardware overhead expense profiles.

## Execution & Automation Flow
The `rag_pipeline.py` script executes the following lifecycle dynamically:
1. Scans target PDF filings inside `raw_data/` page by page.
2. Uses keyword filters to isolate dense context (Operating Revenue or Risk Factors) and gets rid of irrelevant information.
3. Formulates structured local responses, tracks compute speeds, models operational cost telemetry, and puts data into `rag_execution_log.csv`.
4. Feeds the log into a Tableau dashboard workbook (`RAG_Analytics_Dashboard.twbx`).
