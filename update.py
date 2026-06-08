import subprocess
import sys
import shutil
import os
from datetime import datetime

print(f"=== LUMHS Update Started: {datetime.now()} ===", flush=True)

# Step 1 - Clear old ChromaDB
print("Step 1: Clearing ChromaDB...", flush=True)
if os.path.exists('./chroma_db'):
    shutil.rmtree('./chroma_db')
    print("Cleared.", flush=True)

# Step 2 - Scrape fresh data
print("Step 2: Scraping website...", flush=True)
subprocess.run([sys.executable, 'scraper.py'], check=True)
print("Scraping done.", flush=True)

# Step 3 - Re-embed
print("Step 3: Embedding data...", flush=True)
subprocess.run([sys.executable, 'embed.py'], check=True)
print("Embedding done.", flush=True)

print(f"=== Update Complete: {datetime.now()} ===", flush=True)