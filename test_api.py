import asyncio
import aiohttp
import json
from datetime import datetime

async def test_dexscreener_endpoints():
    """Test different DexScreener API endpoints"""
    
    endpoints = [
        # Different variations of DexScreener endpoints
        "https://api.dexscreener.com/latest/dex/search?q=SOL",
        "https://api.dexscreener.com/latest/dex/tokens/solana/So11111111111111111111111111111111111111112",
        "https://api.dexscreener.com/latest/dex/pairs/solana/8HoQnePLqPj4M7PUDzfw8e3Ymdwgc7NLGnaTUapubyvu",  # SOL-USDC pair
        "https://api.dexscreener.com/latest/dex/pairs/solana",
        "https://api.dexscreener.com/latest/dex/tokens/SOL"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json'
    }

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                print(f"\nTesting endpoint: {endpoint}")
                async with session.get(endpoint, headers=headers) as response:
                    print(f"Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        if 'pairs' in data:
                            print(f"Found {len(data['pairs'])} pairs")
                            if data['pairs']:
                                print("\nSample data:")
                                print(json.dumps(data['pairs'][0], indent=2))
                    else:
                        print(f"Error: {await response.text()}")
            
            except Exception as e:
                print(f"Error with endpoint: {str(e)}")
            
            # Respect rate limits
            await asyncio.sleep(1)

async def main():
    print(f"Starting DexScreener API tests at {datetime.now()}")
    await test_dexscreener_endpoints()
    print("\nTests complete!")

if __name__ == "__main__":
    asyncio.run(main())