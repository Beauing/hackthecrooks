import aiohttp
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_second: int = 2):
        self.calls_per_second = calls_per_second
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    async def wait(self):
        """Wait if needed to respect rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.minimum_interval:
            wait_time = self.minimum_interval - time_since_last_call
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()

class TokenScanner:
    def __init__(self):
        self.selection_criteria = {
            'min_liquidity_usd': 100000,  # Minimum $100k liquidity
            'min_volume_24h': 50000,      # Minimum $50k daily volume
            'max_price_impact': 0.02,     # Maximum 2% price impact
            'min_holders': 100,           # Minimum number of holders
            'min_age_days': 7,            # Minimum token age in days
            'suspicious_patterns': [       # Patterns to avoid
                'honeypot',
                'high_tax',
                'locked_liquidity'
            ]
        }
        
        # Initialize rate limiters for different APIs
        self.dexscreener_limiter = RateLimiter(2)  # 2 calls per second
        self.solscan_limiter = RateLimiter(3)      # 3 calls per second
        
        # API endpoints
        self.api_endpoints = {
            'primary': {
                'dexscreener': "https://api.dexscreener.com/latest/dex/pairs/solana",
                'solscan': "https://api.solscan.io/token/meta"
            },
            'backup': {
                'birdeye': "https://public-api.birdeye.so/public/tokens",
                'jupiter': "https://price.jup.ag/v4/price"
            }
        }
        
        # Track API health
        self.api_health = {endpoint: True for endpoint in self.api_endpoints['primary']}
        self.last_api_error = {}

    async def make_api_call(self, url: str, api_name: str, timeout: int = 10) -> Optional[Dict]:
        """Make API call with error handling and rate limiting"""
        try:
            # Apply rate limiting
            if api_name == 'dexscreener':
                await self.dexscreener_limiter.wait()
            elif api_name == 'solscan':
                await self.solscan_limiter.wait()

            timeout_obj = ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        self.api_health[api_name] = True
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        logger.warning(f"{api_name} rate limit hit, backing off")
                        await asyncio.sleep(5)  # Back off for 5 seconds
                        return None
                    else:
                        logger.error(f"{api_name} API error: Status {response.status}")
                        self.api_health[api_name] = False
                        self.last_api_error[api_name] = {
                            'time': datetime.now(),
                            'status': response.status
                        }
                        return None

        except asyncio.TimeoutError:
            logger.error(f"{api_name} API timeout")
            self.api_health[api_name] = False
            return None
        except Exception as e:
            logger.error(f"{api_name} API error: {str(e)}")
            self.api_health[api_name] = False
            return None

    async def fetch_all_pairs(self) -> List[Dict]:
        """Fetch all Solana token pairs with fallback options"""
        # Try primary API (DexScreener)
        pairs = await self.make_api_call(
            self.api_endpoints['primary']['dexscreener'],
            'dexscreener'
        )
        
        if pairs and 'pairs' in pairs:
            logger.info(f"Successfully fetched {len(pairs['pairs'])} pairs from DexScreener")
            return pairs.get('pairs', [])
            
        # Try backup API (Birdeye)
        logger.warning("Primary API failed, trying backup")
        backup_pairs = await self.make_api_call(
            self.api_endpoints['backup']['birdeye'],
            'birdeye'
        )
        
        if backup_pairs and 'data' in backup_pairs:
            logger.info(f"Successfully fetched {len(backup_pairs['data'])} pairs from backup API")
            # Convert backup API format to match primary
            return self.convert_backup_pairs(backup_pairs['data'])
            
        logger.error("All APIs failed to fetch pairs")
        return []

    def convert_backup_pairs(self, backup_data: List[Dict]) -> List[Dict]:
        """Convert backup API data format to match primary API"""
        converted_pairs = []
        for item in backup_data:
            try:
                converted_pair = {
                    'pairAddress': item.get('address'),
                    'baseToken': {
                        'address': item.get('address'),
                        'name': item.get('symbol'),
                        'symbol': item.get('symbol')
                    },
                    'liquidity': {
                        'usd': item.get('liquidity', 0)
                    },
                    'volume': {
                        'h24': item.get('volume24h', 0)
                    },
                    'priceUsd': str(item.get('price', 0))
                }
                converted_pairs.append(converted_pair)
            except Exception as e:
                logger.error(f"Error converting pair data: {str(e)}")
                continue
                
        return converted_pairs

    async def fetch_token_metrics(self, token_address: str) -> Dict:
        """Fetch token metrics with multiple sources and retries"""
        metrics = {
            'holder_count': 0,
            'creation_date': None,
            'is_verified': False
        }

        try:
            url = f"{self.api_endpoints['primary']['solscan']}?token={token_address}"
            data = await self.make_api_call(url, 'solscan')
            
            if data:
                metrics.update({
                    'holder_count': data.get('holder_count', 0),
                    'creation_date': data.get('creation_date'),
                    'is_verified': data.get('is_verified', False)
                })
                return metrics
                
            # Try backup sources if primary fails
            logger.warning(f"Primary metrics source failed for {token_address}, trying backup")
            backup_metrics = await self.fetch_backup_metrics(token_address)
            metrics.update(backup_metrics or {})
            
        except Exception as e:
            logger.error(f"Error fetching token metrics: {str(e)}")
            
        return metrics

    async def fetch_backup_metrics(self, token_address: str) -> Optional[Dict]:
        """Fetch metrics from backup sources"""
        try:
            jupiter_url = f"{self.api_endpoints['backup']['jupiter']}?id={token_address}"
            jupiter_data = await self.make_api_call(jupiter_url, 'jupiter')
            
            if jupiter_data:
                return {
                    'holder_count': jupiter_data.get('holders', 0),
                    'is_verified': jupiter_data.get('verified', False)
                }
                
        except Exception as e:
            logger.error(f"Error fetching backup metrics: {str(e)}")
            
        return None

    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(24)  # Annualized hourly volatility

    def calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate momentum score based on recent price action"""
        if len(prices) < 24:  # Need at least 24 hours of data
            return 0
            
        returns = np.diff(prices) / prices[:-1]
        weighted_returns = np.average(returns, weights=np.linspace(0, 1, len(returns)))
        return weighted_returns

    async def analyze_pair(self, pair: Dict) -> Dict:
        """Analyze trading pair with enhanced error handling"""
        try:
            logger.info(f"Starting analysis for pair: {pair}")  # Log the input pair data
            
            # Validate input
            if not isinstance(pair, dict):
                logger.error(f"Invalid pair format: Expected dict, got {type(pair)}")
                return None
                
            if 'baseToken' not in pair:
                logger.error("Missing baseToken in pair data")
                return None

            # Handle native SOL case first
            is_sol = pair.get('baseToken', {}).get('address') == 'So11111111111111111111111111111111111111112'
            if is_sol:
                logger.info("Processing native SOL token")
                return {
                    'pair_address': pair.get('pairAddress', 'SOL-PAIR'),
                    'token_address': 'So11111111111111111111111111111111111111112',
                    'token_name': 'SOL',
                    'liquidity': float(pair.get('liquidity', {}).get('usd', 1000000)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 500000)),
                    'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'volatility': 0.1,  # Default moderate volatility for SOL
                    'momentum': 0.0,
                    'holder_count': 1000000,
                    'is_verified': True,
                    'score': 80  # High default score for SOL
                }

            # For other tokens
            try:
                liquidity = float(pair.get('liquidity', {}).get('usd', 0))
                logger.info(f"Liquidity: {liquidity}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error parsing liquidity: {e}")
                liquidity = 0

            try:
                volume_24h = float(pair.get('volume', {}).get('h24', 0))
                logger.info(f"Volume 24h: {volume_24h}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error parsing volume: {e}")
                volume_24h = 0

            try:
                price_change_24h = float(pair.get('priceChange', {}).get('h24', 0))
                logger.info(f"Price change 24h: {price_change_24h}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error parsing price change: {e}")
                price_change_24h = 0

            # Log metrics before fetching
            logger.info("Fetching token metrics...")
            metrics = await self.fetch_token_metrics(pair['baseToken']['address'])
            logger.info(f"Fetched metrics: {metrics}")

            # Safe price history handling
            try:
                price_history = pair.get('price_history', [])
                if not isinstance(price_history, list) or not price_history:
                    logger.info("No price history found, using current price")
                    current_price = float(pair.get('priceUsd', 0))
                    price_history = [current_price]
            except Exception as e:
                logger.error(f"Error handling price history: {e}")
                price_history = [0.0]

            logger.info(f"Price history length: {len(price_history)}")

            # Calculate metrics with detailed logging
            try:
                volatility = self.calculate_volatility(price_history)
                momentum = self.calculate_momentum_score(price_history)
                logger.info(f"Calculated volatility: {volatility}, momentum: {momentum}")
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                volatility = 0.0
                momentum = 0.0

            analysis = {
                'pair_address': pair.get('pairAddress', ''),
                'token_address': pair['baseToken']['address'],
                'token_name': pair['baseToken'].get('name', 'Unknown'),
                'liquidity': liquidity,
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'volatility': volatility,
                'momentum': momentum,
                'holder_count': metrics.get('holder_count', 0),
                'is_verified': metrics.get('is_verified', False),
                'score': 0
            }

            logger.info(f"Completed analysis successfully: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Critical error in analyze_pair: {str(e)}")
            logger.error(f"Pair data causing error: {pair}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def calculate_opportunity_score(self, analysis: Dict) -> float:
        """Calculate overall opportunity score"""
        try:
            score = 0
            
            # Liquidity score (0-20 points)
            liq_score = min(20, analysis['liquidity'] / 100000)
            score += liq_score
            
            # Volume score (0-20 points)
            vol_score = min(20, analysis['volume_24h'] / 50000)
            score += vol_score
            
            # Momentum score (0-20 points)
            mom_score = max(0, min(20, analysis['momentum'] * 100))
            score += mom_score
            
            # Volatility score (0-20 points)
            # Prefer moderate volatility (10-30%)
            vol = analysis['volatility'] * 100
            if 10 <= vol <= 30:
                vol_score = 20
            else:
                vol_score = max(0, 20 - abs(vol - 20))
            score += vol_score
            
            # Holder count score (0-20 points)
            holder_score = min(20, analysis['holder_count'] / 100)
            score += holder_score
            
            # Normalize to 0-100
            return score
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0

    async def scan_for_opportunities(self, max_pairs: int = 5) -> List[Dict]:
        """Scan for trading opportunities with enhanced error handling"""
        opportunities = []
        error_count = 0
        max_errors = 3
        
        # Fetch all pairs
        pairs = await self.fetch_all_pairs()
        
        if not pairs:
            logger.error("Failed to fetch any pairs")
            return []
            
        logger.info(f"Starting analysis of {len(pairs)} pairs")
        
        # Analyze each pair
        for pair in pairs:
            try:
                # Basic filtering
                if float(pair.get('liquidity', {}).get('usd', 0)) < self.selection_criteria['min_liquidity_usd']:
                    continue
                    
                if float(pair.get('volume', {}).get('h24', 0)) < self.selection_criteria['min_volume_24h']:
                    continue
                
                # Detailed analysis
                analysis = await self.analyze_pair(pair)
                if not analysis:
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error("Too many analysis errors, aborting scan")
                        break
                    continue
                
                # Calculate opportunity score
                analysis['score'] = self.calculate_opportunity_score(analysis)
                
                # Add to opportunities if score is good enough
                if analysis['score'] >= 60:
                    opportunities.append(analysis)
                    logger.info(f"Found promising opportunity: {analysis['token_name']} (Score: {analysis['score']})")
            
            except Exception as e:
                logger.error(f"Error processing pair: {str(e)}")
                error_count += 1
                if error_count >= max_errors:
                    break
        
        # Sort by score and return top opportunities
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities[:max_pairs]

    def is_suspicious(self, pair: Dict) -> bool:
        """Enhanced suspicious pattern detection"""
        try:
            red_flags = 0
            reasons = []
            
            # Check liquidity ratio
            if pair['liquidity'] > 0:
                volume_liq_ratio = pair['volume_24h'] / pair['liquidity']
                if volume_liq_ratio > 5:
                    red_flags += 1
                    reasons.append(f"High volume/liquidity ratio: {volume_liq_ratio:.2f}")
            
            # Check price movements
            if abs(pair['price_change_24h']) > 50:
                red_flags += 1
                reasons.append(f"Extreme price change: {pair['price_change_24h']:.2f}%")
            
            # Check volatility
            if pair['volatility'] > 0.5:
                red_flags += 1
                reasons.append(f"High volatility: {pair['volatility']:.2f}")
            
            if red_flags >= 2:
                logger.warning(f"Suspicious pair detected: {reasons}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in suspicious check: {str(e)}")
            return True  # Err on the side of caution

    async def get_api_status(self) -> Dict:
        """Get current status of all APIs"""
        return {
            'api_health': self.api_health,
            'last_errors': self.last_api_error,
            'timestamp': datetime.now().isoformat()
        }