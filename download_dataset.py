"""
Robust auto-retry downloader for UBFC-Phys dataset.
Automatically retries on connection failures — kagglehub resumes
from the last downloaded byte, so no data is lost.
"""

import time
import kagglehub

DATASET = "phanquythinh/ubfc-phys-s1-s14"
MAX_RETRIES = 50
RETRY_DELAY = 10  # seconds

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n{'='*50}")
    print(f"  Attempt {attempt}/{MAX_RETRIES}")
    print(f"{'='*50}")
    try:
        path = kagglehub.dataset_download(DATASET)
        print(f"\n✅ Download COMPLETE!")
        print(f"   Path: {path}")
        break
    except Exception as e:
        print(f"\n❌ Failed: {type(e).__name__}: {e}")
        if attempt < MAX_RETRIES:
            print(f"   Retrying in {RETRY_DELAY}s (will resume from last byte)...")
            time.sleep(RETRY_DELAY)
        else:
            print("   Max retries reached. Please try again later.")
