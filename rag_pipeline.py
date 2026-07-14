import os
import time
import pandas as pd
import random
from pypdf import PdfReader

# =====================================================================
# 1. SETUP & CONFIGURATION (NO API KEY REQUIRED!)
# =====================================================================

# Define our 4 test questions, which files they need to read, and who they belong to
test_queries = [
    {
        "id": 1, 
        "company": "United Airlines", 
        "pdf_path": "raw_data/united_10k.pdf", 
        "q": "What was United's operating revenue in FY2024?"
    },
    {
        "id": 2, 
        "company": "Delta Air Lines", 
        "pdf_path": "raw_data/delta_10k.pdf", 
        "q": "What was Delta's operating revenue in FY2024?"
    },
    {
        "id": 3, 
        "company": "United Airlines", 
        "pdf_path": "raw_data/united_10k.pdf", 
        "q": "What risk factors impacted operational expenses, pilot shortages, or fuel prices?"
    },
    {
        "id": 4, 
        "company": "Delta Air Lines", 
        "pdf_path": "raw_data/delta_10k.pdf", 
        "q": "What were the primary risk factors mentioned regarding supply chain delays or aircraft?"
    }
]

# =====================================================================
# 2. HELPER FUNCTION: PARSE THE RAW PDFs (Data Engineering Layer)
# =====================================================================
def extract_relevant_pdf_text(pdf_path, keywords):
    """
    Opens a raw PDF file, reads page by page, and extracts text only from
    pages containing our target keywords (e.g., 'revenue' or 'risk').
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find '{pdf_path}'. Make sure you have a 'raw_data' folder with your PDFs.")
        return ""
    
    print(f"📖 Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    extracted_text = []
    
    # Check each page for matching keywords to filter out irrelevant text
    for page in reader.pages:
        text = page.extract_text() or ""
        if any(keyword.lower() in text.lower() for keyword in keywords):
            extracted_text.append(text)
            
    # Combine the matched pages and cap it to form a standard data window
    full_context = "\n".join(extracted_text)
    return full_context


# =====================================================================
# 3. RUNNING THE PIPELINE (Deterministic Local NLP Inference Simulator)
# =====================================================================
performance_logs = []

print("🚀 Starting Local Unsupervised NLP & Analytics Pipeline...")

for query in test_queries:
    print(f"\nProcessing Question ID {query['id']} ({query['company']})...")
    
    # Step A: Pick search keywords based on what the question is asking
    if "revenue" in query["q"].lower():
        search_keywords = ["revenue", "operating revenue", "passenger revenues", "operating income"]
    else:
        search_keywords = ["risk factors", "supply chain", "fuel prices", "pilot shortage"]
        
    # Step B: Extract matching text using our PDF helper function
    context_text = extract_relevant_pdf_text(query["pdf_path"], search_keywords)
    
    if not context_text:
        print(f"⚠️ Warning: No context could be extracted from {query['pdf_path']}. Skipping.")
        continue
        
    # Step C: Start a timer to measure Latency (Data Science Metric #1)
    start_time = time.time()
    
    # Step D: Local Text Extraction Mechanics (Simulating Natural Language Pruning)
    # The script looks for lines containing keywords to form a deterministic answer snippet
    lines = context_text.split('\n')
    matched_insights = []
    for line in lines:
        if any(kw.lower() in line.lower() for kw in search_keywords):
            if len(line.strip()) > 30 and line.strip() not in matched_insights:
                matched_insights.append(line.strip())
        if len(matched_insights) >= 3: # Keep the top 3 high-density context insights
            break
            
    # Formulate a structured text response from the PDF without cloud dependencies
    if matched_insights:
        ai_response = "Extracted Financial Context: " + " | ".join(matched_insights)
    else:
        ai_response = "Keyword focus identified, but dense string sequence was out of bounding bounds."

    # Artificial constraint layer to mimic computing latency based on text size
    compute_delay = len(context_text) / 50000.0 + random.uniform(0.2, 0.5)
    time.sleep(compute_delay)
    
    # Step E: Stop the timer
    end_time = time.time()
    latency = round(end_time - start_time, 2)
    
    # Step F: Track token/word metrics (Data Science Metric #2)
    # 1 token roughly equals 0.75 words. We derive input tokens directly from the text volume!
    input_tokens = int(len(context_text.split()) / 0.75)
    output_tokens = int(len(ai_response.split()) / 0.75)
    
    # Step G: Compute simulated infrastructure operational expenses (Data Science Metric #3)
    # We measure local CPU machine cost baseline rather than cloud API pricing metrics
    local_hardware_cost_per_token = 0.0000012 
    estimated_cost = (input_tokens + output_tokens) * local_hardware_cost_per_token
    
    # Log everything into our execution dataset array
    performance_logs.append({
        "Question_ID": query["id"],
        "Target_Company": query["company"],
        "User_Question": query["q"],
        "AI_Response": ai_response,
        "Latency_Seconds": latency,
        "Input_Tokens": input_tokens,
        "Output_Tokens": output_tokens,
        "Run_Cost_USD": estimated_cost
    })
    
    print(f"✅ Question {query['id']} complete! Latency: {latency}s | Computed Log Cost: ${estimated_cost:.5f}")

# =====================================================================
# 4. SAVE EXPORT TO CSV FOR TABLEAU
# =====================================================================
output_df = pd.DataFrame(performance_logs)
output_df.to_csv("rag_execution_log.csv", index=False)

print("\n📊 Process Complete! Your file has been saved as: rag_execution_log.csv")
print("You can now connect this file directly to Tableau Desktop.")