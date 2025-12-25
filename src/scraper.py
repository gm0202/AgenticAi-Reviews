import json
import os
import datetime
from google_play_scraper import Sort, reviews
from datetime import datetime, timedelta

DATA_DIR = "data"

def fetch_daily_reviews(app_id: str, target_date_str: str, country: str = 'in', lang: str = 'en'):
    """
    Fetches reviews for a specific date. 
    Notes:
    - google-play-scraper doesn't allow filtering by exact date in the query.
    - We must fetch 'Newest' reviews and safeguard until we pass the target date.
    - This is a simplified approach assuming we run this daily or for a recent range.
    """
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    
    # Ensure data dir exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    file_path = os.path.join(DATA_DIR, f"{target_date_str}.json")
    
    # Check cache
    if os.path.exists(file_path):
        print(f"[CACHE HIT] Data for {target_date_str} already exists.")
        return

    print(f"[FETCHING] Reviews for {target_date_str}...")
    
    # We fetch a buffer to ensure we cover the whole day. 
    # Since we can't filter by date API-side, we iterate until we see a date OLDER than target.
    all_reviews = []
    continuation_token = None
    
    # Safety break to avoid infinite loops if something goes wrong
    MAX_QUERIES = 100 
    
    query_count = 0
    done = False
    
    while not done and query_count < MAX_QUERIES:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=200, # Max per request
            continuation_token=continuation_token
        )
        
        query_count += 1
        
        if not result:
            break
            
        for r in result:
            review_date = r['at'].date()
            
            if review_date == target_date:
                all_reviews.append({
                    'reviewId': r['reviewId'],
                    'content': r['content'],
                    'score': r['score'],
                    'at': r['at'].isoformat()
                })
            elif review_date < target_date:
                # We have gone past the target date
                done = True
                pass # Don't break immediately, we might have mixed dates in the batch? (Usually sorted)
                
        # If the last review in this batch is older than target_date, we can stop strictly
        if result[-1]['at'].date() < target_date:
            done = True

    # Save to disk
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=2)
    
    print(f"[SAVED] {len(all_reviews)} reviews for {target_date_str} to {file_path}")

def batch_scrape(app_id, start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    curr = start_date
    while curr <= end_date:
        fetch_daily_reviews(app_id, curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

if __name__ == "__main__":
    # Test with Swiggy for a recent date
    # Swiggy App ID: in.swiggy.android
    # Zomato App ID: com.application.zomato
    APP_ID = "in.swiggy.android"
    
    # Fetch yesterday's data primarily
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Testing scraper for {yesterday}")
    fetch_daily_reviews(APP_ID, yesterday)
