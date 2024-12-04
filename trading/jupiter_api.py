import aiohttp
import json
from typing import List, Dict, Optional
import asyncio
from datetime import datetime

class JupiterAPI:
    def __init__(self):
        self.base_url = "https://price.jup.ag/v4"
        self.quote_url = "https://quote-api.jup.ag/v4"
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        # Common token addresses
        self.tokens = {
            'SOL': 'So11111111111111111111111111111111111111112',
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263'
        }

    async def get_price(self, input_mint: str, output_mint: str) -> Optional[Dict]:
        """Get real-time price data"""
        try:
            url = f"{self.base_url}/price?ids={input_mint}&vsToken={output_mint}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {})
                    else:
                        print(f"Price API error: Status {response.status}")
                        return None
        except Exception as e:
            print(f"Error fetching price: {str(e)}")
            return None

    async def get_quote(self, 
                       input_mint: str, 
                       output_mint: str, 
                       amount: int, 
                       slippage: float = 0.5) -> Optional[Dict]:
        """Get quote for swap"""
        try:
            url = f"{self.quote_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": int(slippage * 100)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Quote API error: Status {response.status}")
                        return None
        except Exception as e:
            print(f"Error fetching quote: {str(e)}")
            return None

    async def get_token_list(self) -> List[Dict]:
        """Get list of supported tokens"""
        try:
            url = "https://token.jup.ag/all"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Token list API error: Status {response.status}")
                        return []
        except Exception as e:
            print(f"Error fetching token list: {str(e)}")
            return []

    async def monitor_price(self, 
                          input_mint: str, 
                          output_mint: str, 
                          interval: int = 5):
        """Monitor price in real-time"""
        while True:
            price_data = await self.get_price(input_mint, output_mint)
            if price_data:
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                print(f"Price: ${float(price_data.get('price', 0)):.4f}")
                print("-" * 40)
            await asyncio.sleep(interval)

# Test functionality
async def test_jupiter():
    api = JupiterAPI()
    
    print("Testing Jupiter API...")
    
    # Test price fetching
    print("\nTesting price fetch for SOL/USDC:")
    price_data = await api.get_price(api.tokens['SOL'], api.tokens['USDC'])
    if price_data:
        print(json.dumps(price_data, indent=2))
    
    # Test quote
    print("\nTesting quote for 1 SOL to USDC:")
    quote = await api.get_quote(
        api.tokens['SOL'],
        api.tokens['USDC'],
        1_000_000_000  # 1 SOL in lamports
    )
    if quote:
        print(json.dumps(quote, indent=2))
    
    # Test token list
    print("\nFetching supported tokens:")
    tokens = await api.get_token_list()
    if tokens:
        print(f"Found {len(tokens)} supported tokens")
        print("\nSample tokens:")
        for token in tokens[:3]:
            print(f"- {token.get('symbol')}: {token.get('address')}")

    return api

if __name__ == "__main__":
    asyncio.run(test_jupiter())