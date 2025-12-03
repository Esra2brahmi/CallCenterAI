# Not rcommended
"""
Generate training data for DistilGPT-2 router by querying both services
and analyzing their responses to determine optimal routing
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Tuple
import logging
from langdetect import detect, LangDetectException
import re
from textstat import flesch_reading_ease, syllable_count
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Service URLs
# ==============================
TFIDF_SERVICE_URL = "http://127.0.0.1:8000/predict"
TRANSFORMER_SERVICE_URL = "http://0.0.0.0:8001/predict"

# ==============================
# Feature Extraction
# ==============================
def extract_text_features(text: str) -> Dict:
    """Extract comprehensive features from text for routing decision"""
    
    # Basic metrics
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(re.split(r'[.!?]+', text))
    avg_word_length = char_count / max(word_count, 1)
    
    # Language detection
    try:
        language = detect(text)
    except LangDetectException:
        language = "unknown"
    
    # Complexity metrics
    try:
        readability = flesch_reading_ease(text)
    except:
        readability = 50.0  # default medium complexity
    
    # Vocabulary richness (unique words / total words)
    words = text.lower().split()
    vocabulary_richness = len(set(words)) / max(len(words), 1)
    
    # Special characters and numbers
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(char_count, 1)
    number_ratio = len(re.findall(r'\d', text)) / max(char_count, 1)
    
    # Punctuation density
    punctuation_count = len(re.findall(r'[.!?,;:]', text))
    punctuation_density = punctuation_count / max(sentence_count, 1)
    
    # Capital letters ratio
    capital_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    
    # Email and phone indicators (PII complexity)
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'\b(?:\+?\d{1,3})?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b', text))
    
    # Named entity indicators (simplified)
    capitalized_words = len([w for w in words if w and w[0].isupper()])
    proper_noun_ratio = capitalized_words / max(word_count, 1)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'language': language,
        'readability': readability,
        'vocabulary_richness': vocabulary_richness,
        'special_char_ratio': special_char_ratio,
        'number_ratio': number_ratio,
        'punctuation_density': punctuation_density,
        'capital_ratio': capital_ratio,
        'has_email': int(has_email),
        'has_phone': int(has_phone),
        'proper_noun_ratio': proper_noun_ratio
    }

# ==============================
# Query Services
# ==============================
def query_tfidf_service(text: str, timeout: int = 5) -> Dict:
    """Query TF-IDF service and get prediction + confidence"""
    try:
        response = requests.post(
            TFIDF_SERVICE_URL,
            json={"text": text},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract confidence (max probability)
        probs = data.get('probability', {})
        max_confidence = max(probs.values()) if probs else 0.0
        
        return {
            'success': True,
            'prediction': data.get('label', 'unknown'),
            'confidence': max_confidence,
            'probabilities': probs,
            'latency': response.elapsed.total_seconds()
        }
    except Exception as e:
        logger.warning(f"TF-IDF service error: {e}")
        return {
            'success': False,
            'confidence': 0.0,
            'error': str(e)
        }

def query_transformer_service(text: str, timeout: int = 10) -> Dict:
    """Query Transformer service and get prediction + confidence"""
    try:
        response = requests.post(
            TRANSFORMER_SERVICE_URL,
            json={"text": text},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            'success': True,
            'prediction': data.get('prediction', 'unknown'),
            'confidence': data.get('confidence', 0.0),
            'probabilities': data.get('probabilities', {}),
            'latency': response.elapsed.total_seconds()
        }
    except Exception as e:
        logger.warning(f"Transformer service error: {e}")
        return {
            'success': False,
            'confidence': 0.0,
            'error': str(e)
        }

# ==============================
# Routing Decision Logic
# ==============================
def determine_optimal_route(text: str, features: Dict, 
                           tfidf_result: Dict, 
                           transformer_result: Dict) -> Tuple[int, str]:
    """
    Determine which service should handle this request
    Returns: (label, reason)
        label: 0 = TF-IDF, 1 = Transformer
        reason: explanation for the decision
    """
    
    reasons = []
    use_transformer = False
    
    # Rule 1: TF-IDF confidence
    tfidf_confidence = tfidf_result.get('confidence', 0.0)
    if tfidf_confidence < 0.75:
        use_transformer = True
        reasons.append(f"Low TF-IDF confidence ({tfidf_confidence:.2f})")
    
    # Rule 2: Text length
    word_count = features['word_count']
    if word_count > 100:
        use_transformer = True
        reasons.append(f"Long text ({word_count} words)")
    
    # Rule 3: Language support
    language = features['language']
    if language not in ['en', 'fr']:
        use_transformer = True
        reasons.append(f"Unsupported language ({language})")
    
    # Rule 4: Text complexity (readability)
    readability = features['readability']
    if readability < 30:  # Very difficult text
        use_transformer = True
        reasons.append(f"Complex text (readability: {readability:.1f})")
    
    # Rule 5: High vocabulary richness (diverse vocabulary)
    vocab_richness = features['vocabulary_richness']
    if vocab_richness > 0.7:
        use_transformer = True
        reasons.append(f"Rich vocabulary ({vocab_richness:.2f})")
    
    # Rule 6: Many special characters or technical content
    special_ratio = features['special_char_ratio']
    number_ratio = features['number_ratio']
    if special_ratio > 0.15 or number_ratio > 0.15:
        use_transformer = True
        reasons.append(f"Technical content (special: {special_ratio:.2f}, nums: {number_ratio:.2f})")
    
    # Rule 7: High proper noun density (named entities)
    proper_noun_ratio = features['proper_noun_ratio']
    if proper_noun_ratio > 0.3:
        use_transformer = True
        reasons.append(f"Many named entities ({proper_noun_ratio:.2f})")
    
    # Rule 8: Service failure
    if not tfidf_result.get('success', False):
        use_transformer = True
        reasons.append("TF-IDF service unavailable")
    
    # Rule 9: Performance comparison (if both succeeded)
    if tfidf_result.get('success') and transformer_result.get('success'):
        # If predictions differ significantly, prefer transformer for complex cases
        if tfidf_result['prediction'] != transformer_result['prediction']:
            transformer_conf = transformer_result.get('confidence', 0.0)
            if transformer_conf > tfidf_confidence + 0.1:
                use_transformer = True
                reasons.append(f"Transformer more confident ({transformer_conf:.2f} vs {tfidf_confidence:.2f})")
    
    label = 1 if use_transformer else 0
    reason = "; ".join(reasons) if reasons else "Default to TF-IDF (simple, confident case)"
    
    return label, reason

# ==============================
# Generate Training Dataset
# ==============================
def generate_training_data(input_csv: str, output_csv: str, sample_size: int = None):
    """
    Generate training data by querying both services
    
    Args:
        input_csv: Path to CSV with 'text' column
        output_csv: Path to save training data
        sample_size: Optional sample size for testing
    """
    
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info(f"Sampling {len(df)} records for training data generation")
    
    logger.info(f"Processing {len(df)} texts...")
    
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processed {idx}/{len(df)} texts...")
        
        text = str(row['Document'])
        
        # Extract features
        features = extract_text_features(text)
        
        # Query both services
        tfidf_result = query_tfidf_service(text)
        time.sleep(0.1)  # Rate limiting
        
        transformer_result = query_transformer_service(text)
        time.sleep(0.1)  # Rate limiting
        
        # Determine optimal route
        route_label, reason = determine_optimal_route(
            text, features, tfidf_result, transformer_result
        )
        
        # Compile result
        result = {
            'Document': text,
            'router_label': route_label,
            'reason': reason,
            
            # Features
            **features,
            
            # TF-IDF results
            'tfidf_confidence': tfidf_result.get('confidence', 0.0),
            'tfidf_prediction': tfidf_result.get('prediction', 'error'),
            'tfidf_success': tfidf_result.get('success', False),
            'tfidf_latency': tfidf_result.get('latency', 0.0),
            
            # Transformer results
            'transformer_confidence': transformer_result.get('confidence', 0.0),
            'transformer_prediction': transformer_result.get('prediction', 'error'),
            'transformer_success': transformer_result.get('success', False),
            'transformer_latency': transformer_result.get('latency', 0.0),
        }
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    logger.info(f"✅ Training data saved to {output_csv}")
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    print(f"Total samples: {len(results_df)}")
    print(f"TF-IDF routing: {(results_df['router_label'] == 0).sum()} ({(results_df['router_label'] == 0).sum()/len(results_df)*100:.1f}%)")
    print(f"Transformer routing: {(results_df['router_label'] == 1).sum()} ({(results_df['router_label'] == 1).sum()/len(results_df)*100:.1f}%)")
    
    print("\nFeature Statistics:")
    print(f"Avg word count: {results_df['word_count'].mean():.1f}")
    print(f"Avg TF-IDF confidence: {results_df['tfidf_confidence'].mean():.2f}")
    print(f"Avg Transformer confidence: {results_df['transformer_confidence'].mean():.2f}")
    
    print("\nLanguage distribution:")
    print(results_df['language'].value_counts().head(5))
    
    print("\nTop reasons for Transformer routing:")
    transformer_reasons = results_df[results_df['router_label'] == 1]['reason']
    for reason in transformer_reasons.value_counts().head(5).index:
        count = transformer_reasons.value_counts()[reason]
        print(f"  - {reason}: {count} times")
    
    return results_df

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    import argparse
    
    # Hardcoded paths
    input_csv = "data/processed/sample.csv"
    output_csv = "router_training_data.csv"
    sample_size = None  # or set an integer if you want


    
    # Check if services are running
    logger.info("Checking service availability...")
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=2)
        logger.info("✅ TF-IDF service is running")
    except:
        logger.error("❌ TF-IDF service not available at http://127.0.0.1:8000")
        exit(1)
    
    try:
        requests.get("http://localhost:8001/health", timeout=2)
        logger.info("✅ Transformer service is running")
    except:
        logger.error("❌ Transformer service not available at http://localhost:8001")
        exit(1)
    
    # Generate training data
    generate_training_data(input_csv, output_csv, sample_size)