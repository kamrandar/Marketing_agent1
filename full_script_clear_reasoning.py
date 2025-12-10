import pandas as pd
import time
import os
import threading

from google.genai import types
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from langchain_community.utilities import RequestsWrapper


# ========================================================
# 0. Timeout Wrapper (Windows-Compatible)
# ========================================================
def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return "TIMEOUT"
    if exception[0]:
        raise exception[0]

    return result[0]


# ========================================================
# 1. Setup
# ========================================================
google_search_tool = Tool(google_search=GoogleSearch())

requests_wrapper = RequestsWrapper(
    headers={"User-Agent": "Mozilla/5.0"}
)

from goog import GOOGLE_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)

CSV_PATH = r"C:\Users\KamranImtiyaz\OneDrive - MinoriLabs\Desktop\Agent\New Ecom - Industry Segmentation _ Oct2025(WIP).csv"
OUTPUT_PATH = r"classified_output.csv"

# -------------------------
# Load from previous run if available (preserve old labels)
# -------------------------
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing output file -> loading from {OUTPUT_PATH} to preserve previous labels.")
    df = pd.read_csv(OUTPUT_PATH)
else:
    df = pd.read_csv(CSV_PATH)

# Ensure column exists
if "Category_Gemini" not in df.columns:
    df["Category_Gemini"] = ""

# 🔥 Normalization to make skip logic robust (trims whitespace and normalizes typical nulls)
df["Category_Gemini"] = df["Category_Gemini"].astype(str).str.strip()
df["Category_Gemini"].replace({"nan": "", "NaN": "", "None": "", "none": ""}, inplace=True)


# ========================================================
# 2. Manual Start Options
# ========================================================
MANUAL_START = True
MANUAL_ROW = 0

if MANUAL_START:
    start_index = MANUAL_ROW
    print(f"Manual start enabled → Starting at row {start_index}")
else:
    # Find first truly empty row in the (possibly previously-saved) dataframe
    empty_mask = df["Category_Gemini"].isna() | (df["Category_Gemini"].astype(str).str.strip() == "")
    if empty_mask.any():
        start_index = empty_mask.idxmax()
    else:
        start_index = None
    print(f"Auto-resume enabled → Starting at first unclassified row: {start_index}")

if start_index is None or pd.isna(start_index):
    print("✔ All rows already classified!")
    exit()


# ========================================================
# 3. Fetch Website HTML
# ========================================================
def fetch_website_html(url):
    if pd.isna(url) or str(url).strip() == "":
        return "NO_URL"

    try:
        html = requests_wrapper.get(url)
        if not html:
            return "NO_HTML"
        return html
    except Exception as e:
        return f"ERROR_FETCHING_HTML: {e}"


# ========================================================
# 4. Gemini Website Classifier
# ========================================================
def classify_with_gemini(url):
    if pd.isna(url) or str(url).strip() == "":
        return "other"

    html_content = fetch_website_html(url)
    html_trimmed = html_content[:5000] if isinstance(html_content, str) else str(html_content)[:5000]

    prompt = (
        f"You are an expert website classifier. Your task is to analyze websites and categorize them accurately. "
        f"Analyze the website at {url}.\n\n"
        f"Here is the website's HTML content:\n"
        f"---------------- HTML START ----------------\n"
        f"{html_trimmed}\n"
        f"---------------- HTML END ----------------\n\n"
        "Using BOTH the HTML and Google Search information, classify the website into one of the following categories: "
        "1) 'e-commerce' — if it primarily enables online sales or transactions and includes features such as product listings, shopping carts, checkout or payment gateways, customer accounts, promotional offers, or order tracking. "
        "2) 'mark-ops' — if it focuses on marketing, branding, lead generation, or corporate communications, typically featuring sections like 'About Us', 'Services', 'Case Studies', 'Blog', or 'Contact Us' but without direct purchase options. "
        "3) 'other' — if it does not clearly fit into either category or its a nan or you don't get any thing from Google search.\n\n"
        "Respond with only: 'e-commerce', 'mark-ops', or 'other' nothing else."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            thinking_config=types.ThinkingConfig(
                thinking_budget=20000,
                include_thoughts=True
            ),
        )
    )

    print("\n---------------- GEMINI RAW OUTPUT ----------------")
    print(response)
    print("---------------------------------------------------\n")

    if response is None or getattr(response, "text", None) is None:
        return "other"

    return response.text.strip().lower()


# ========================================================
# 5. Batch Processing
# ========================================================
BATCH_SIZE = 15
TOTAL = len(df)

print(f"Total rows: {TOTAL}")
print(f"Processing from row {start_index}...\n")


# ========================================================
# 6. MAIN LOOP (FINAL SKIP-FIXED VERSION)
# ========================================================
for batch_start in range(start_index, TOTAL, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, TOTAL)

    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"\n==============================")
    print(f"PROCESSING BATCH {batch_start} → {batch_end - 1}")
    print(f"==============================\n")

    for i in range(batch_start, batch_end):

        old_value = str(df.at[i, "Category_Gemini"]).strip().lower()

        # --------------------------------------
        # FINAL SKIP LOGIC (will never call Gemini unnecessarily)
        # --------------------------------------
        if old_value not in ["", "nan", "none"]:
            print(f"⏩ Row {i} already classified ({old_value}) → Skipping")
            continue

        url = df.at[i, "Website"]
        print(f"\n🔵 Row {i} — URL: {url}")

        try:
            category = run_with_timeout(classify_with_gemini, args=(url,), timeout=20)

            if category == "TIMEOUT":
                print("⏱️ Timeout reached — skipping this URL")
                category = "timeout"

        except Exception as e:
            print(f"❌ Error: {e}")
            category = "other"

        df.at[i, "Category_Gemini"] = category
        print(f"✅ Classified as: {category}\n")

    # Save progress to OUTPUT_PATH so future runs preserve these values
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Progress saved to → {OUTPUT_PATH}")

    print("\n⏳ Sleeping for 30 seconds...\n")
    time.sleep(30)

print("\n🎉 ALL DONE — Entire CSV classified successfully!")
