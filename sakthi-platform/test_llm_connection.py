import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Test your actual LLM servers
endpoints = [
    "http://10.100.15.67:1138/v1/chat/completions",  # DeepSeek Fast
    "http://10.100.15.67:1137/v1/chat/completions"   # SQLCoder Accurate
]

for endpoint in endpoints:
    try:
        print(f"Testing: {endpoint}")
        
        response = requests.post(
            endpoint,
            json={
                "model": "deepseek-1.3b-q5" if "1138" in endpoint else "sqlcoder-7b-sql",
                "messages": [
                    {"role": "user", "content": "Convert Oracle procedures to BigQuery"}
                ],
                "max_tokens": 100,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ SUCCESS - {endpoint}")
            result = response.json()
            print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')[:100]}...")
        else:
            print(f"❌ HTTP {response.status_code} - {endpoint}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ CONNECTION ERROR - {endpoint}")
        print(f"Error: {str(e)}")
    
    print("-" * 60)