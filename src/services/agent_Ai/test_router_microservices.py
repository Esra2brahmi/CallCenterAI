"""
Interactive testing tool for Agent AI router service
"""

import requests
import json
import time
from typing import Dict, List
import pandas as pd
from tabulate import tabulate
import argparse

# ==============================
# Configuration
# ==============================
AGENT_AI_URL = "http://localhost:8000"
TFIDF_URL = "http://localhost:8002"
TRANSFORMER_URL = "http://localhost:8001"

# ==============================
# Test Cases
# ==============================
TEST_CASES = [
    # Simple cases (should route to TF-IDF)
    {
        "text": "I need to reset my password",
        "expected_service": "TF-IDF",
        "category": "Simple query"
    },
    {
        "text": "What are your business hours?",
        "expected_service": "TF-IDF",
        "category": "Simple question"
    },
    {
        "text": "I want to cancel my subscription",
        "expected_service": "TF-IDF",
        "category": "Common request"
    },
    
    # Complex cases (should route to Transformer)
    {
        "text": """I am experiencing a critical issue with our enterprise deployment where 
        the microservices architecture is failing to properly handle distributed transactions 
        across multiple database shards, leading to data inconsistency and potential ACID 
        violations. We need immediate assistance with implementing a two-phase commit protocol 
        or alternative consensus mechanism that can ensure data integrity while maintaining 
        the required throughput of 10,000 transactions per second.""",
        "expected_service": "Transformer",
        "category": "Long technical text"
    },
    {
        "text": """Je souhaiterais obtenir des informations détaillées concernant les modalités 
        de résiliation anticipée de mon contrat, les éventuelles pénalités associées, ainsi 
        que les démarches administratives nécessaires pour transférer mes données personnelles 
        conformément aux dispositions du RGPD.""",
        "expected_service": "Transformer",
        "category": "French complex query"
    },
    {
        "text": "这是一个中文测试文本，用于验证系统是否能正确处理非英语和法语的内容。",
        "expected_service": "Transformer",
        "category": "Unsupported language (Chinese)"
    },
    
    # Edge cases
    {
        "text": "Help! My account was hacked and I see unauthorized transactions!!!",
        "expected_service": "TF-IDF",
        "category": "Urgent but simple"
    },
    {
        "text": "$$$$ CLICK HERE FOR FREE MONEY $$$$ Contact: scam@example.com or call +1-555-SCAM",
        "expected_service": "Transformer",
        "category": "Spam/PII heavy"
    },
]

# ==============================
# Helper Functions
# ==============================
def check_service_health(url: str, service_name: str) -> bool:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"⚠️  {service_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} is unavailable: {e}")
        return False

def test_prediction(text: str) -> Dict:
    """Test a single prediction"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{AGENT_AI_URL}/predict",
            json={"text": text},
            timeout=30
        )
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "prediction": data.get("prediction"),
                "confidence": data.get("confidence"),
                "model_used": data.get("model_used"),
                "router_decision": data.get("router_decision", {}),
                "latency": latency
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}",
                "latency": latency
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": 0
        }

def test_route_decision_only(text: str) -> Dict:
    """Test routing decision without calling downstream services"""
    try:
        response = requests.post(
            f"{AGENT_AI_URL}/route_decision",
            json={"text": text},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ==============================
# Test Suites
# ==============================
def run_health_checks():
    """Run health checks on all services"""
    print("\n" + "="*60)
    print("HEALTH CHECKS")
    print("="*60)
    
    services = [
        (AGENT_AI_URL, "Agent AI Router"),
        (TFIDF_URL, "TF-IDF Service"),
        (TRANSFORMER_URL, "Transformer Service")
    ]
    
    all_healthy = True
    for url, name in services:
        if not check_service_health(url, name):
            all_healthy = False
    
    if all_healthy:
        print("\n✅ All services are healthy!\n")
    else:
        print("\n⚠️  Some services are unavailable. Tests may fail.\n")
    
    return all_healthy

def run_predefined_tests():
    """Run predefined test cases"""
    print("\n" + "="*60)
    print("PREDEFINED TEST CASES")
    print("="*60)
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {test_case['category']}")
        print(f"Text: {test_case['text'][:80]}...")
        
        result = test_prediction(test_case['text'])
        
        if result["success"]:
            router_service = result["router_decision"].get("service", "Unknown")
            router_conf = result["router_decision"].get("confidence", 0.0)
            reasoning = result["router_decision"].get("reasoning", "")
            
            match = "✅" if router_service == test_case["expected_service"] else "❌"
            
            print(f"{match} Routed to: {router_service} (confidence: {router_conf:.2%})")
            print(f"   Expected: {test_case['expected_service']}")
            print(f"   Reasoning: {reasoning}")
            print(f"   Prediction: {result['prediction']} ({result['confidence']:.2%})")
            print(f"   Latency: {result['latency']:.3f}s")
            
            results.append({
                "Category": test_case["category"],
                "Expected": test_case["expected_service"],
                "Actual": router_service,
                "Match": match,
                "Confidence": f"{router_conf:.2%}",
                "Latency": f"{result['latency']:.3f}s"
            })
        else:
            print(f"❌ Error: {result['error']}")
            results.append({
                "Category": test_case["category"],
                "Expected": test_case["expected_service"],
                "Actual": "ERROR",
                "Match": "❌",
                "Confidence": "N/A",
                "Latency": "N/A"
            })
    
    # Summary table
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Statistics
    matches = sum(1 for r in results if r["Match"] == "✅")
    total = len(results)
    print(f"\nAccuracy: {matches}/{total} ({matches/total*100:.1f}%)")

def interactive_mode():
    """Interactive testing mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter text to test routing (or 'quit' to exit)\n")
    
    while True:
        text = input("\n📝 Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        print("\n🔄 Testing routing decision...")
        route_result = test_route_decision_only(text)
        
        if "error" not in route_result:
            print(f"\n{'='*50}")
            print(f"🎯 Router Decision: {route_result.get('decision')}")
            print(f"📊 Confidence: {route_result.get('confidence', 0):.2%}")
            print(f"💭 Reasoning: {route_result.get('reasoning')}")
            print(f"⏱️  Latency: {route_result.get('latency', 0):.3f}s")
            print(f"{'='*50}")
            
            # Ask if user wants full prediction
            full_predict = input("\n🔮 Run full prediction? (y/n): ").strip().lower()
            if full_predict == 'y':
                print("\n🔄 Running full prediction...")
                pred_result = test_prediction(text)
                
                if pred_result["success"]:
                    print(f"\n{'='*50}")
                    print(f"🏷️  Prediction: {pred_result['prediction']}")
                    print(f"📊 Confidence: {pred_result['confidence']:.2%}")
                    print(f"🤖 Model Used: {pred_result['model_used']}")
                    print(f"⏱️  Total Latency: {pred_result['latency']:.3f}s")
                    print(f"{'='*50}")
                else:
                    print(f"❌ Error: {pred_result['error']}")
        else:
            print(f"❌ Error: {route_result['error']}")

def benchmark_mode(num_requests: int = 100):
    """Benchmark routing performance"""
    print("\n" + "="*60)
    print(f"BENCHMARK MODE ({num_requests} requests)")
    print("="*60)
    
    # Use predefined test cases in rotation
    texts = [tc["text"] for tc in TEST_CASES]
    
    latencies = []
    router_latencies = []
    service_latencies = []
    tfidf_count = 0
    transformer_count = 0
    
    print(f"\nSending {num_requests} requests...")
    
    for i in range(num_requests):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_requests}")
        
        text = texts[i % len(texts)]
        result = test_prediction(text)
        
        if result["success"]:
            latencies.append(result["latency"])
            router_lat = result["router_decision"].get("latency", 0)
            service_lat = result["router_decision"].get("service_latency", 0)
            router_latencies.append(router_lat)
            service_latencies.append(service_lat)
            
            if result["router_decision"].get("service") == "TF-IDF":
                tfidf_count += 1
            else:
                transformer_count += 1
    
    # Statistics
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n📊 Request Distribution:")
    print(f"   TF-IDF: {tfidf_count} ({tfidf_count/num_requests*100:.1f}%)")
    print(f"   Transformer: {transformer_count} ({transformer_count/num_requests*100:.1f}%)")
    
    print(f"\n⏱️  Latency Statistics:")
    print(f"   Total Latency:")
    print(f"      Mean: {sum(latencies)/len(latencies)*1000:.2f}ms")
    print(f"      Median: {sorted(latencies)[len(latencies)//2]*1000:.2f}ms")
    print(f"      Min: {min(latencies)*1000:.2f}ms")
    print(f"      Max: {max(latencies)*1000:.2f}ms")
    
    print(f"\n   Router Latency:")
    print(f"      Mean: {sum(router_latencies)/len(router_latencies)*1000:.2f}ms")
    
    print(f"\n   Service Latency:")
    print(f"      Mean: {sum(service_latencies)/len(service_latencies)*1000:.2f}ms")
    
    print(f"\n🚀 Throughput: {num_requests/sum(latencies):.2f} requests/sec")

def get_router_stats():
    """Get current router statistics"""
    try:
        response = requests.get(f"{AGENT_AI_URL}/router_stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("\n" + "="*60)
            print("ROUTER STATISTICS")
            print("="*60)
            print(f"Total Requests: {stats['total_requests']}")
            print(f"TF-IDF Routes: {stats['tfidf_routes']} ({stats['tfidf_percentage']:.1f}%)")
            print(f"Transformer Routes: {stats['transformer_routes']} ({stats['transformer_percentage']:.1f}%)")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Test Agent AI Router Service')
    parser.add_argument('--health', action='store_true', help='Run health checks only')
    parser.add_argument('--test', action='store_true', help='Run predefined test cases')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    parser.add_argument('--benchmark', type=int, metavar='N', help='Benchmark with N requests')
    parser.add_argument('--stats', action='store_true', help='Show router statistics')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AGENT AI ROUTER - TESTING TOOL")
    print("="*60)
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run selected tests
    if args.health or args.all:
        run_health_checks()
    
    if args.test or args.all:
        run_predefined_tests()
    
    if args.benchmark:
        benchmark_mode(args.benchmark)
    
    if args.stats or args.all:
        get_router_stats()
    
    if args.interactive:
        interactive_mode()

if __name__ == "__main__":
    main()