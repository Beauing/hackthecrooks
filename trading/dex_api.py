import aiohttp
import json
from typing import List, Dict
import asyncio

class DexScreenerAPI:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }

    async def get_pairs(self, search_term: str = "SOL/USDC") -> List[Dict]:
        """Get trading pairs using search"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}?q={search_term}"
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        # Filter for valid pairs
                        filtered_pairs = []
                        for pair in pairs:
                            if self.validate_pair(pair):
                                filtered_pairs.append(self.format_pair(pair))
                        
                        return filtered_pairs
                    else:
                        print(f"Error: Status {response.status}")
                        return []
        except Exception as e:
            print(f"Error fetching pairs: {str(e)}")
            return []

    def validate_pair(self, pair: Dict) -> bool:
        """Validate pair data"""
        required_fields = [
            'chainId', 'pairAddress', 'baseToken', 'quoteToken',
            'priceUsd', 'volume', 'liquidity'
        ]
        
        if not all(field in pair for field in required_fields):
            return False
            
        # Ensure it's a Solana pair
        if pair['chainId'] != 'solana':
            return False
            
        # Check for minimum liquidity ($10k)
        if float(pair.get('liquidity', {}).get('usd', 0)) < 10000:
            return False
            
        return True

    def format_pair(self, pair: Dict) -> Dict:
        """Format pair data for our use"""
        return {
            'pair_address': pair['pairAddress'],
            'base_token': {
                'address': pair['baseToken']['address'],
                'symbol': pair['baseToken']['symbol'],
                'name': pair['baseToken']['name']
            },
            'quote_token': {
                'address': pair['quoteToken']['address'],
                'symbol': pair['quoteToken']['symbol']
            },
            'price_usd': float(pair['priceUsd']),
            'liquidity_usd': float(pair['liquidity']['usd']),
            'volume_24h': float(pair['volume'].get('h24', 0)),
            'price_change': {
                'h1': float(pair['priceChange'].get('h1', 0)),
                'h24': float(pair['priceChange'].get('h24', 0))
            },
            'transactions_24h': {
                'buys': pair['txns'].get('h24', {}).get('buys', 0),
                'sells': pair['txns'].get('h24', {}).get('sells', 0)
            }
        }

# Test the API
async def test_api():
    api = DexScreenerAPI()
    
    print("Testing DexScreener API...")
    pairs = await api.get_pairs()
    
    print(f"\nFound {len(pairs)} valid pairs")
    if pairs:
        print("\nSample pair data:")
        print(json.dumps(pairs[0], indent=2))
    
    return pairs

if __name__ == "__main__":
    asyncio.run(test_api())