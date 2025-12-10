import pandas as pd
import time
from google.genai import types
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# ================================
# 1. Setup
# ================================
google_search_tool = Tool(google_search=GoogleSearch())

from goog import GOOGLE_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)

CSV_PATH = r"C:\Users\KamranImtiyaz\OneDrive - MinoriLabs\Desktop\Agent\New Ecom - Industry Segmentation _ Oct2025(WIP).csv"
OUTPUT_PATH = r"classified_output.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# If Category column doesn't exist, create it
if "Category_Gemini" not in df.columns:
    df["Category_Gemini"] = ""

# ================================
# 2. Gemini Website Classifier
# ================================
def classify_with_gemini(url):
    if pd.isna(url) or url.strip() == "":
        return "other"

    prompt = (
        f"Analyze the website at {url}. "
        "Classify the website into one of: 'e-commerce', 'mark-ops', or 'other'. "
        "Return only the category label."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            thinking_config=types.ThinkingConfig(
                thinking_budget=5000,
                include_thoughts=False
            ),
        )
    )

    # Protection in case Gemini returns empty
    if response is None or getattr(response, "text", None) is None:
        return "other"

    return response.text.strip().lower()


# ================================
# 3. Batch Processing Logic
# ================================
BATCH_SIZE = 15
TOTAL = len(df)

print(f"Total rows to classify: {TOTAL}")

# Find starting point (resume support)
start_index = df[df["Category_Gemini"] == ""].index.min()

if pd.isna(start_index):
    print("All rows already classified!")
    exit()

print(f"Starting classification from row: {start_index}")

# ================================
# 4. Main Loop
# ================================
for batch_start in range(start_index, TOTAL, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, TOTAL)
    print(f"\n=== Classifying rows {batch_start} to {batch_end - 1} ===")

    for i in range(batch_start, batch_end):
        url = df.at[i, "Website"]
        print(f"Row {i} — URL: {url}")

        try:
            category = classify_with_gemini(url)
        except Exception as e:
            print(f"Error for row {i}: {e}")
            category = "other"

        df.at[i, "Category_Gemini"] = category
        print(f" → Classified as: {category}")

    # Save progress after every batch
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Progress saved to {OUTPUT_PATH}")

    # Sleep 2 minutes before next batch
    print("Sleeping for 2 minutes...\n")
    time.sleep(120)   # 120 seconds pause

print("\n### ALL DONE! Classification finished. ###")
