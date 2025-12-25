import json
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
try:
    from visualization import generate_visualizations
except ImportError:
    # Fallback if running from root without module
    from src.visualization import generate_visualizations

# Constants
TAXONOMY_FILE = "taxonomy.json"
DATA_DIR = "data"
OUTPUT_DIR = "output"
SIMILARITY_THRESHOLD = 0.82

# --- 1. TAXONOMY & EMBEDDING MANAGER ---
class TaxonomyManager:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.taxonomy_path = TAXONOMY_FILE
        self.topics = {} # {topic_name: {examples: [], embedding: np.array, created_at: str}}
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.load_taxonomy()

    def load_taxonomy(self):
        if os.path.exists(self.taxonomy_path):
            with open(self.taxonomy_path, "r") as f:
                data = json.load(f)
                # Restore numpy arrays from lists
                for topic, details in data.items():
                    details["embedding"] = np.array(details["embedding"])
                self.topics = data
            print(f"[TAXONOMY] Loaded {len(self.topics)} topics.")
        else:
            print("[TAXONOMY] No existing taxonomy found. Starting fresh.")
            self.topics = {}

    def save_taxonomy(self):
        # Convert numpy arrays to lists for JSON serialization
        serializable_topics = {}
        for topic, details in self.topics.items():
            serializable_topics[topic] = {
                "examples": details["examples"],
                "embedding": details["embedding"].tolist(),
                "created_at": details["created_at"]
            }
        
        # Backup before write
        if os.path.exists(self.taxonomy_path):
            shutil.copy(self.taxonomy_path, self.taxonomy_path + ".bak")
            
        with open(self.taxonomy_path, "w") as f:
            json.dump(serializable_topics, f, indent=2)

    def get_topic_embedding(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text))

    def map_extracted_topic(self, raw_topic: str) -> str:
        """
        Matches a raw extracted topic string to an existing taxonomy topic.
        Returns the existing topic name if match found, else returns the raw_topic (which implies it's new).
        """
        if not self.topics:
            return raw_topic
        
        raw_embedding = self.get_topic_embedding(raw_topic).reshape(1, -1)
        
        # Calculate similarity with all existing topics
        existing_topics = list(self.topics.keys())
        existing_embeddings = np.array([self.topics[t]["embedding"] for t in existing_topics])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(raw_embedding, existing_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_topic = existing_topics[best_idx]
        
        print(f"[DEBUG] '{raw_topic}' vs '{best_topic}' score: {best_score:.3f}")
        
        if best_score >= SIMILARITY_THRESHOLD:
            return best_topic
        else:
            return raw_topic

    def add_new_topic(self, topic_name: str):
        if topic_name not in self.topics:
            emb = self.get_topic_embedding(topic_name)
            self.topics[topic_name] = {
                "examples": [topic_name],
                "embedding": emb,
                "created_at": datetime.now().isoformat()
            }
            pass # We only save periodically or at end of batch to avoid too much IO

# --- 2. REVIEW EXTRACTION AGENT (GROQ) ---
class ReviewAgent:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )
        self.parser = JsonOutputParser()
        
        # Prompt for extraction
        self.extract_prompt = ChatPromptTemplate.from_template("""
        You are an expert user researcher. Your task is to analyze a batch of App Store reviews and exact specific issues, requests, or feedback points.
        
        For each review, if it contains a clear issue, bug report, feature request, or specific praise/complaint, extract it as a short, concise topic string (3-6 words).
        Ignore generic reviews like "Good", "Nice", "Worst app" unless they specify WHY.
        
        Reviews:
        {reviews_text}
        
        Return the output as a JSON List of objects:
        [
            {{ "reviewId": "...", "topic": "..." }},
            ...
        ]
        If a review has no specific content, do not include it in the list.
        """)
        
        self.chain = self.extract_prompt | self.llm | self.parser

    def extract_topics_batch(self, reviews_batch: List[Dict]) -> List[Dict]:
        """
        Takes a list of raw review dicts and returns topic extractions.
        """
        # Format reviews for prompt
        reviews_text = ""
        for r in reviews_batch:
            # Filter extremely short reviews to save tokens and noise
            if len(r['content']) < 4: 
                continue
            reviews_text += f"ID: {r['reviewId']}\nText: {r['content']}\n---\n"
            
        if not reviews_text:
            return []

        try:
            result = self.chain.invoke({"reviews_text": reviews_text})
            return result
        except Exception as e:
            print(f"[ERROR] LLM Extraction failed: {e}")
            return []

# --- 3. PIPELINE ORCHESTRATOR ---
def process_daily_batch(date_str: str, taxonomy_mgr: TaxonomyManager, agent: ReviewAgent):
    file_path = os.path.join(DATA_DIR, f"{date_str}.json")
    if not os.path.exists(file_path):
        print(f"[SKIP] No data for {date_str}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)
    
    print(f"[PROCESSING] {len(reviews)} reviews for {date_str}")
    
    daily_topics = {} # {topic: count}
    
    # Process in chunks of 20 to avoid context limits
    CHUNK_SIZE = 20
    for i in range(0, len(reviews), CHUNK_SIZE):
        batch = reviews[i:i+CHUNK_SIZE]
        
        # 1. Extract
        extracted_items = agent.extract_topics_batch(batch)
        
        # 2. Map & Count
        for item in extracted_items:
            raw_topic = item.get("topic")
            if not raw_topic: continue
            
            # Map valid topics
            mapped_topic = taxonomy_mgr.map_extracted_topic(raw_topic)
            
            # Update taxonomy if new
            if mapped_topic not in taxonomy_mgr.topics:
                print(f"[NEW TOPIC] {mapped_topic}")
                taxonomy_mgr.add_new_topic(mapped_topic)
            
            # Count
            daily_topics[mapped_topic] = daily_topics.get(mapped_topic, 0) + 1
            
        print(f"   Processed batch {i//CHUNK_SIZE + 1}/{(len(reviews)//CHUNK_SIZE)+1}")

    # Save daily stats
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    stats_file = os.path.join(OUTPUT_DIR, f"stats_{date_str}.json")
    with open(stats_file, "w") as f:
        json.dump(daily_topics, f, indent=2)
        
    # Persist taxonomy updates
    taxonomy_mgr.save_taxonomy()
    print(f"[DONE] Daily stats saved to {stats_file}")

    # Generate Graphs and Heatmaps
    generate_visualizations(OUTPUT_DIR, date_str)

if __name__ == "__main__":
    # Test run
    # Expects GROQ_API_KEY env var
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("Please set GROQ_API_KEY environment variable.")
        exit(1)
        
    print("Initializing Agent...")
    tax_mgr = TaxonomyManager()
    agent = ReviewAgent(key)
    
    # Run for the date we scraped
    test_date = "2025-12-24" 
    process_daily_batch(test_date, tax_mgr, agent)
