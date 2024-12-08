import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

async def fetch_trading_pairs(self):
    """Find potential trading pairs with better error handling"""
    self.log_activity("Scanning for trading pairs", "Scanning")
    try:
        # First try DexScreener V2 API
        urls = [
            "https://api.dexscreener.com/latest/dex/tokens/solana",
            "https://api.dexscreener.com/latest/dex/search?q=SOL",
            "https://public-api.birdeye.so/public/tokenlist?sort_by=v24h_desc"  # Backup API
        ]
        
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    self.log_activity(f"Trying API endpoint: {url}", "API")
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Handle different API response formats
                            pairs = []
                            if 'pairs' in data:
                                pairs = data['pairs']
                            elif 'data' in data:
                                pairs = data['data']
                                
                            if pairs:
                                self.log_activity(f"Successfully found {len(pairs)} pairs")
                                return self.filter_promising_pairs(pairs)
                        
                        self.log_activity(f"API returned status {response.status}")
                
                except Exception as e:
                    self.log_activity(f"Error with endpoint {url}: {str(e)}")
                    continue
            
            # If we get here, all APIs failed
            self.log_activity("All API endpoints failed, using backup pairs", "Fallback")
            return self.get_backup_pairs()
                    
    except Exception as e:
        self.log_activity(f"Critical error in pair fetching: {str(e)}", "Error")
        return self.get_backup_pairs()

def get_backup_pairs(self):
    """Return backup trading pairs when APIs fail"""
    backup_pairs = [
        {
            'pairAddress': 'So11111111111111111111111111111111111111112',
            'baseToken': {'symbol': 'SOL', 'address': 'So11111111111111111111111111111111111111112'},
            'quoteToken': {'symbol': 'USDC', 'address': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'},
            'priceUsd': '100',  # Will be updated with real price
            'liquidity': {'usd': '1000000'},
            'volume': {'h24': '500000'}
        },
        # Add more backup pairs here
    ]
    
    self.log_activity(f"Using {len(backup_pairs)} backup pairs", "Fallback")
    return backup_pairs

def filter_promising_pairs(self, pairs):
    """Filter pairs based on criteria"""
    good_pairs = []
    for pair in pairs:
        try:
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            volume = float(pair.get('volume', {}).get('h24', 0))
            
            if liquidity > 100000 and volume > 50000:
                good_pairs.append(pair)
                self.log_activity(
                    f"Added pair {pair.get('baseToken', {}).get('symbol', 'Unknown')}"
                    f"/{pair.get('quoteToken', {}).get('symbol', 'Unknown')}"
                )
                
        except Exception as e:
            continue
            
    self.log_activity(f"Found {len(good_pairs)} promising pairs after filtering")
    return good_pairs