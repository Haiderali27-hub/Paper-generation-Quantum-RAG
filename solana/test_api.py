"""Test API endpoints to see what data we get"""
import asyncio
import aiohttp
import json

async def test_endpoint(url):
    print(f"\n{'='*80}")
    print(f"Testing: {url}")
    print('='*80)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as resp:
                print(f"Status: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    print(f"Response type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"Keys: {list(data.keys())}")
                        if 'pairs' in data:
                            print(f"Number of pairs: {len(data['pairs'])}")
                            if data['pairs']:
                                print(f"First pair sample: {json.dumps(data['pairs'][0], indent=2)[:500]}")
                        if 'data' in data:
                            print(f"Data content: {json.dumps(data['data'], indent=2)[:500]}")
                    elif isinstance(data, list):
                        print(f"List length: {len(data)}")
                        if data:
                            print(f"First item: {json.dumps(data[0], indent=2)[:500]}")
                else:
                    text = await resp.text()
                    print(f"Error response: {text[:300]}")
    except Exception as e:
        print(f"Exception: {str(e)}")

async def main():
    urls = [
        "https://api.dexscreener.com/token-profiles/latest/v1",
        "https://api.dexscreener.com/token-boosts/latest/v1",
    ]
    
    for url in urls:
        await test_endpoint(url)
        await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(main())
