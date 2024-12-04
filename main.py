from trading.strategies import AdvancedTradingStrategy, TradingSignal
from trading.risk_manager import RiskManager, Position
from trading.token_analysis import EnhancedTokenAnalysis
from trading.pattern_detection import SuspiciousPatternDetector
from trading.token_scanner import TokenScanner
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import asyncio
import json
import base58
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
import aiohttp
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
import uvicorn
import pandas as pd

class EnhancedTradingBot:
    def __init__(self):
        self.strategy = AdvancedTradingStrategy()
        self.risk_manager = RiskManager()
        self.token_analyzer = EnhancedTokenAnalysis()
        self.pattern_detector = SuspiciousPatternDetector()
        self.token_scanner = TokenScanner()
        self.client = AsyncClient("https://api.mainnet-beta.solana.com")
        
        # Trading state
        self.trading_state = {
            'is_trading': False,
            'current_pair': None,
            'chart_data': [],
            'wallet_key': None,
            'positions': [],
            'active_connections': set(),
            'analyzed_tokens': {},
            'blacklisted_tokens': set(),
            'pending_trades': {},
            'trade_history': []
        }
        
        # Configuration
        self.config = {
            'min_safety_score': 0.7,        # Minimum token safety score
            'scan_interval': 3600,          # Token scanning interval (1 hour)
            'analysis_window': 24 * 3600,   # Analysis window (24 hours)
            'max_positions': 5,             # Maximum concurrent positions
            'alert_threshold': 0.4,         # Alert threshold for risk score
            'auto_blacklist_threshold': 0.2, # Auto-blacklist threshold
            'max_slippage': 0.01,          # Maximum allowed slippage (1%)
            'min_liquidity': 50000,        # Minimum liquidity in USD
            'gas_limit': 300000,           # Maximum gas limit for transactions
            'retry_attempts': 3,           # Number of retry attempts for failed transactions
            'min_profit_threshold': 0.02    # Minimum profit threshold (2%)
        }
        
        self.last_scan_time = datetime.now()
        
        # Initialize price feeds
        self.price_feeds = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'average_trade_duration': 0,
            'win_rate': 0
        }

    async def analyze_token(self, token_address: str) -> Dict:
        """Analyze token with special handling for SOL"""
        try:
            # Special case for SOL
            if token_address == 'So11111111111111111111111111111111111111112':
                return {
                    'safety_score': 1.0,  # Highest safety for SOL
                    'metrics': {
                        'liquidity': 1000000,
                        'volume_24h': 500000,
                        'holders': 1000000,
                    },
                    'token_name': 'SOL',
                    'is_verified': True,
                    'warnings': []
                }

            # Normal token analysis
            token_data = await self.fetch_token_data(token_address)
            if not token_data:
                logger.error(f"No token data found for {token_address}")
                return None

            # Calculate metrics
            metrics = {
                'liquidity': float(token_data.get('liquidity', {}).get('usd', 0)),
                'volume_24h': float(token_data.get('volume', {}).get('h24', 0)),
                'holders': token_data.get('holders', 0),
                'price_change_24h': float(token_data.get('priceChange', {}).get('h24', 0))
            }

            # Calculate safety score
            safety_factors = {
                'sufficient_liquidity': metrics['liquidity'] > 100000,
                'sufficient_volume': metrics['volume_24h'] > 50000,
                'enough_holders': metrics['holders'] > 100,
                'stable_price': abs(metrics['price_change_24h']) < 30
            }

            safety_score = sum(1 for factor in safety_factors.values() if factor) / len(safety_factors)

            return {
                'safety_score': safety_score,
                'metrics': metrics,
                'warnings': self.generate_warnings(metrics, safety_factors)
            }

        except Exception as e:
            logger.error(f"Error analyzing token {token_address}: {str(e)}")
            return None

    def generate_warnings(self, metrics: Dict, safety_factors: Dict) -> List[str]:
        """Generate warning messages based on metrics"""
        warnings = []
        
        if not safety_factors['sufficient_liquidity']:
            warnings.append(f"Low liquidity: ${metrics['liquidity']:.2f}")
        if not safety_factors['sufficient_volume']:
            warnings.append(f"Low volume: ${metrics['volume_24h']:.2f}")
        if not safety_factors['enough_holders']:
            warnings.append(f"Few holders: {metrics['holders']}")
        if not safety_factors['stable_price']:
            warnings.append(f"High volatility: {abs(metrics['price_change_24h'])}%")

        return warnings

    async def fetch_dexscreener_data(self, pair_address: str) -> Dict:
        """Fetch pair data from DexScreener with better error handling"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('pairs', [None])[0]
        except Exception as e:
            print(f"DexScreener API error: {str(e)}")
            return None

        return None

    async def get_jupiter_quote(self, input_mint: str, output_mint: str, 
                              amount: int, slippage: float) -> Dict:
        """Get quote from Jupiter aggregator with error handling"""
        try:
            url = "https://quote-api.jup.ag/v4/quote"
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
                        print(f"Jupiter quote error: {response.status}")
                        return None
        except Exception as e:
            print(f"Error getting Jupiter quote: {str(e)}")
            return None

    async def execute_jupiter_swap(self, quote: Dict) -> Dict:
        """Execute swap on Jupiter with advanced error handling"""
        if not self.trading_state['wallet_key']:
            raise Exception("Wallet not connected")
            
        try:
            private_key = self.trading_state['wallet_key']
            keypair = Keypair.from_secret_key(base58.b58decode(private_key))
            
            url = "https://quote-api.jup.ag/v4/swap"
            data = {
                "quoteResponse": quote,
                "userPublicKey": str(keypair.public_key),
                "wrapUnwrapSOL": True
            }
            
            async with aiohttp.ClientSession() as session:
                # Get swap transaction
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to create swap transaction: {response.status}")
                        
                    swap_data = await response.json()
                    transaction = Transaction.deserialize(swap_data['swapTransaction'])
                    
                    # Set compute budget for transaction
                    compute_budget_ix = ComputeBudgetInstruction.set_compute_unit_limit(
                        self.config['gas_limit']
                    )
                    transaction.instructions.insert(0, compute_budget_ix)
                    
                    # Sign and send transaction
                    signed_tx = transaction.sign(keypair)
                    
                    # Send with retry logic
                    for attempt in range(self.config['retry_attempts']):
                        try:
                            result = await self.client.send_transaction(
                                signed_tx,
                                opts=TxOpts(
                                    skip_preflight=False,
                                    preflight_commitment=Commitment.CONFIRMED
                                )
                            )
                            
                            # Wait for confirmation
                            confirmation = await self.client.confirm_transaction(
                                result.value,
                                commitment=Commitment.CONFIRMED
                            )
                            
                            if confirmation:
                                return {
                                    'success': True,
                                    'signature': str(result.value),
                                    'attempt': attempt + 1
                                }
                        except Exception as e:
                            if attempt == self.config['retry_attempts'] - 1:
                                raise
                            await asyncio.sleep(1)
                            continue
                            
            return {
                'success': False,
                'error': 'Failed to execute swap after all attempts'
            }
                
        except Exception as e:
            print(f"Swap execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def fetch_token_data(self, token_address: str) -> Dict:
        """Fetch comprehensive token data with better error handling"""
        try:
            # First try DexScreener API
            base_url = "https://api.dexscreener.com/latest/dex"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/tokens/{token_address}") as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'pairs' in data and len(data['pairs']) > 0:
                            pair = data['pairs'][0]  # Get the most liquid pair
                            return {
                                'price': float(pair.get('priceUsd', 0)),
                                'liquidity': pair.get('liquidity', {'usd': 0}),
                                'volume': pair.get('volume', {'h24': 0}),
                                'priceChange': pair.get('priceChange', {'h24': 0}),
                                'holders': 100,  # Default value as DexScreener doesn't provide holder count
                                'baseToken': pair.get('baseToken', {}),
                                'quoteToken': pair.get('quoteToken', {})
                            }

            # If DexScreener fails, try backup source
            return {
                'price': 0,
                'liquidity': {'usd': 100000},  # Default values
                'volume': {'h24': 50000},
                'priceChange': {'h24': 0},
                'holders': 100,
                'baseToken': {'address': token_address},
                'quoteToken': {'address': 'USDC'}
            }

        except Exception as e:
            print(f"Error fetching token data: {str(e)}")
            return None

    async def fetch_holder_data(self, session: aiohttp.ClientSession, token_address: str) -> List[Dict]:
        """Fetch token holder data"""
        try:
            async with session.get(f"https://public-api.solscan.io/token/holders/{token_address}") as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            print(f"Error fetching holder data: {str(e)}")
            return []

    async def fetch_transaction_history(self, session: aiohttp.ClientSession, token_address: str) -> List[Dict]:
        """Fetch transaction history"""
        try:
            async with session.get(f"https://public-api.solscan.io/token/txs/{token_address}") as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            print(f"Error fetching transaction history: {str(e)}")
            return []

    async def fetch_contract_data(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Fetch contract data"""
        try:
            async with session.get(f"https://public-api.solscan.io/account/{token_address}") as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            print(f"Error fetching contract data: {str(e)}")
            return {}

    async def execute_trade(self, side: str, amount: float, price: float,
                          stop_loss: float = None, take_profit: float = None) -> bool:
        """Execute a trade with enhanced error handling and monitoring"""
        try:
            # Generate unique trade ID
            trade_id = f"trade_{len(self.trading_state['trade_history'])}"
            
            # Record trade start
            trade_start = {
                'id': trade_id,
                'side': side,
                'amount': amount,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'start_time': datetime.now(),
                'status': 'pending'
            }
            
            self.trading_state['pending_trades'][trade_id] = trade_start

            # Fetch pair data
            pair_data = await self.fetch_dexscreener_data(self.trading_state['current_pair'])
            if not pair_data:
                raise Exception("Failed to fetch pair data")

            # Determine input and output tokens
            input_mint = pair_data['baseToken']['address'] if side == "SELL" else "So11111111111111111111111111111111111111112"
            output_mint = "So11111111111111111111111111111111111111112" if side == "SELL" else pair_data['baseToken']['address']
            
            # Calculate amount in lamports
            decimals = int(pair_data['baseToken']['decimals'])
            amount_in = int(amount * (10 ** decimals))
            
            # Get quote
            quote = await self.get_jupiter_quote(
                input_mint, 
                output_mint, 
                amount_in, 
                self.config['max_slippage']
            )
            
            if not quote:
                raise Exception("Failed to get quote")
                
            # Execute swap
            result = await self.execute_jupiter_swap(quote)
            if not result['success']:
                raise Exception(f"Swap execution failed: {result.get('error', 'Unknown error')}")
                
            # Update trade history
            trade_start['status'] = 'completed'
            trade_start['end_time'] = datetime.now()
            trade_start['transaction_hash'] = result['signature']
            
            self.trading_state['trade_history'].append(trade_start)
            del self.trading_state['pending_trades'][trade_id]
            
            # Update performance metrics
            self.update_performance_metrics(trade_start)
                
            return True
            
        except Exception as e:
            print(f"Trade execution error: {str(e)}")
            if trade_id in self.trading_state['pending_trades']:
                self.trading_state['pending_trades'][trade_id]['status'] = 'failed'
                self.trading_state['pending_trades'][trade_id]['error'] = str(e)
                self.trading_state['trade_history'].append(
                    self.trading_state['pending_trades'][trade_id]
                )
                del self.trading_state['pending_trades'][trade_id]
            return False

    def update_performance_metrics(self, trade: Dict):
        """Update performance metrics after trade completion"""
        self.performance_metrics['total_trades'] += 1
        
        if trade['status'] == 'completed':
            self.performance_metrics['successful_trades'] += 1
            
            # Calculate profit/loss
            entry_price = trade['price']
            exit_price = float(trade.get('exit_price', entry_price))
            amount = trade['amount']
            
            pnl = (exit_price - entry_price) * amount if trade['side'] == "BUY" else (entry_price - exit_price) * amount
            
            self.performance_metrics['total_profit_loss'] += pnl
            self.performance_metrics['best_trade'] = max(self.performance_metrics['best_trade'], pnl)
            self.performance_metrics['worst_trade'] = min(self.performance_metrics['worst_trade'], pnl)
            
            # Update win rate
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['successful_trades'] / 
                self.performance_metrics['total_trades']
            ) * 100
            
            # Update average trade duration
            duration = (trade['end_time'] - trade['start_time']).total_seconds()
            current_avg = self.performance_metrics['average_trade_duration']
            n = self.performance_metrics['successful_trades']
            
            self.performance_metrics['average_trade_duration'] = (
                (current_avg * (n - 1) + duration) / n
            )
        else:
            self.performance_metrics['failed_trades'] += 1

    async def scan_for_opportunities(self):
        """Scan for trading opportunities with safety checks"""
        try:
            # Get potential opportunities
            opportunities = await self.token_scanner.scan_for_opportunities()
            
            # Filter based on safety checks
            safe_opportunities = []
            for opp in opportunities:
                if await self.should_trade_token(opp['token_address']):
                    # Add safety metrics to opportunity data
                    opp['safety_metrics'] = self.trading_state['analyzed_tokens'].get(
                        opp['token_address'], {}
                    )
                    safe_opportunities.append(opp)
            
            return safe_opportunities
            
        except Exception as e:
            print(f"Error scanning for opportunities: {str(e)}")
            return []

    async def process_trading_signal(self, data: Dict) -> Dict:
        """Process trading signal with enhanced safety checks"""
        token_address = data.get('token_address')
        
        # Verify token safety
        if not await self.should_trade_token(token_address):
            return {
                'signal': 'HOLD',
                'reason': 'Token safety check failed',
                'warnings': self.trading_state['analyzed_tokens'].get(token_address, {}).get('warnings', [])
            }

        # Extract price data
        prices = [point['price'] for point in self.trading_state['chart_data']]
        volumes = [point.get('volume', 0) for point in self.trading_state['chart_data']]
        highs = [point.get('high', point['price']) for point in self.trading_state['chart_data']]
        lows = [point.get('low', point['price']) for point in self.trading_state['chart_data']]

        # Get trading signal
        signal = self.strategy.get_advanced_signal(prices, volumes, highs, lows)

        # Calculate volatility
        volatility = self.strategy.indicators.calculate_atr(highs, lows, prices)

        # Get wallet balance
        wallet_pubkey = Keypair.from_secret_key(
            base58.b58decode(self.trading_state['wallet_key'])
        ).public_key
        balance = await self.client.get_balance(wallet_pubkey)
        wallet_balance = float(balance.value) / 1e9

        # Validate trade with risk manager
        can_trade = self.risk_manager.validate_trade(
            signal.signal,
            signal.confidence,
            prices[-1],
            wallet_balance,
            volatility / prices[-1]
        )

        if can_trade and signal.signal != "HOLD":
            # Get risk-adjusted parameters
            trade_params = self.risk_manager.get_risk_adjusted_parameters(
                prices[-1],
                wallet_balance,
                signal.confidence,
                volatility / prices[-1]
            )

            try:
                result = await self.execute_trade(
                    signal.signal,
                    trade_params['position_size'],
                    prices[-1],
                    trade_params['stop_loss'],
                    trade_params['take_profit']
                )

                if result:
                    position = Position(
                        entry_price=prices[-1],
                        size=trade_params['position_size'],
                        entry_time=datetime.now(),
                        stop_loss=trade_params['stop_loss'],
                        take_profit=trade_params['take_profit']
                    )
                    self.trading_state['positions'].append(position)

            except Exception as e:
                return {
                    'error': str(e),
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'indicators': signal.indicators
                }

        # Monitor existing positions
        await self.monitor_positions(prices[-1])

        return {
            'signal': signal.signal,
            'confidence': signal.confidence,
            'indicators': signal.indicators,
            'can_trade': can_trade,
            'positions': len(self.trading_state['positions']),
            'safety_score': self.trading_state['analyzed_tokens'].get(token_address, {}).get('safety_score', 0),
            'performance_metrics': self.performance_metrics
        }

    async def monitor_positions(self, current_price: float):
        """Monitor and manage open positions with enhanced risk management"""
        for position in self.trading_state['positions'][:]:
            # Get position status
            exit_check = self.risk_manager.check_position_exit(
                position,
                current_price,
                datetime.now()
            )
            
            if exit_check['should_exit']:
                try:
                    # Calculate actual position value considering slippage
                    expected_value = position.size * current_price
                    min_acceptable = expected_value * (1 - self.config['max_slippage'])
                    
                    # Close position
                    result = await self.execute_trade(
                        "SELL" if position.entry_price < current_price else "BUY",
                        position.size,
                        current_price,
                        min_acceptable=min_acceptable
                    )
                    
                    if result:
                        # Update statistics
                        trade_result = {
                            'type': 'exit',
                            'entry_price': position.entry_price,
                            'exit_price': current_price,
                            'size': position.size,
                            'pnl': exit_check['current_pnl'],
                            'exit_reason': exit_check['exit_reason']
                        }
                        
                        self.risk_manager.update_daily_stats(trade_result)
                        
                        # Remove position
                        self.trading_state['positions'].remove(position)
                        
                        # Broadcast position closure
                        await self.broadcast_update({
                            'type': 'position_closed',
                            'data': trade_result
                        })
                        
                except Exception as e:
                    print(f"Error closing position: {str(e)}")
                    await self.broadcast_update({
                        'type': 'error',
                        'data': {
                            'message': f"Failed to close position: {str(e)}",
                            'position': {
                                'entry_price': position.entry_price,
                                'size': position.size,
                                'current_price': current_price
                            }
                        }
                    })

    async def broadcast_update(self, data: Dict):
        """Broadcast updates to all connected clients with error handling"""
        disconnected_clients = set()
        
        for connection in self.trading_state['active_connections']:
            try:
                await connection.send_json(data)
            except Exception as e:
                print(f"Error broadcasting update: {str(e)}")
                disconnected_clients.add(connection)
                
        # Remove disconnected clients
        self.trading_state['active_connections'] -= disconnected_clients

    async def trading_loop(self):
        """Main trading loop with comprehensive monitoring and safety checks"""
        while self.trading_state['is_trading']:
            try:
                # Periodic opportunity scanning
                if (datetime.now() - self.last_scan_time).seconds >= self.config['scan_interval']:
                    opportunities = await self.scan_for_opportunities()
                    
                    if opportunities:
                        await self.broadcast_update({
                            'type': 'opportunities',
                            'data': opportunities
                        })
                    
                    self.last_scan_time = datetime.now()

                # Current token monitoring
                current_token = self.trading_state['current_pair']
                if current_token:
                    # Safety verification
                    if not await self.should_trade_token(current_token):
                        await self.broadcast_update({
                            'type': 'safety_alert',
                            'data': {
                                'message': f"Safety check failed for {current_token}",
                                'analysis': self.trading_state['analyzed_tokens'].get(current_token, {})
                            }
                        })
                        self.trading_state['is_trading'] = False
                        continue

                    # Fetch and process market data
                    pair_data = await self.fetch_dexscreener_data(current_token)
                    if pair_data:
                        current_price = float(pair_data['priceUsd'])
                        
                        # Update chart data
                        self.trading_state['chart_data'].append({
                            'timestamp': int(datetime.now().timestamp()),
                            'price': current_price,
                            'volume': float(pair_data.get('volume24h', 0)),
                            'high': float(pair_data.get('priceMax24h', current_price)),
                            'low': float(pair_data.get('priceMin24h', current_price))
                        })

                        # Maintain data window
                        cutoff = datetime.now() - timedelta(hours=24)
                        self.trading_state['chart_data'] = [
                            point for point in self.trading_state['chart_data']
                            if datetime.fromtimestamp(point['timestamp']) > cutoff
                        ]

                        # Process trading signal
                        signal_result = await self.process_trading_signal({
                            'price': current_price,
                            'data': pair_data,
                            'token_address': current_token
                        })

                        # Broadcast comprehensive update
                        await self.broadcast_update({
                            'type': 'update',
                            'data': {
                                'price': current_price,
                                'signal': signal_result,
                                'chart_data': self.trading_state['chart_data'],
                                'positions': len(self.trading_state['positions']),
                                'daily_stats': self.risk_manager.daily_stats,
                                'safety_metrics': self.trading_state['analyzed_tokens'].get(current_token, {}),
                                'performance': self.performance_metrics
                            }
                        })

            except Exception as e:
                error_msg = f"Error in trading loop: {str(e)}"
                print(error_msg)
                await self.broadcast_update({
                    'type': 'error',
                    'data': {'message': error_msg}
                })

            await asyncio.sleep(60)

# Initialize FastAPI app and bot
app = FastAPI()
bot = EnhancedTradingBot()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# FastAPI Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main interface"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/connect-wallet")
async def connect_wallet(private_key: str):
    """Connect wallet endpoint with enhanced security"""
    try:
        # Validate private key format
        if len(private_key) != 88 or not all(c in base58.alphabet for c in private_key):
            raise HTTPException(status_code=400, detail="Invalid private key format")
            
        keypair = Keypair.from_secret_key(base58.b58decode(private_key))
        bot.trading_state['wallet_key'] = private_key
        
        # Get initial wallet balance
        balance = await bot.client.get_balance(keypair.public_key)
        
        return {
            "status": "success",
            "public_key": str(keypair.public_key),
            "balance": float(balance.value) / 1e9
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set-token-pair")
async def set_token_pair(pair_address: str):
    """Set token pair with safety checks"""
    try:
        # Verify pair exists and is valid
        pair_data = await bot.fetch_dexscreener_data(pair_address)
        if not pair_data:
            raise HTTPException(status_code=400, detail="Invalid pair address")
            
        # Perform safety analysis
        if not await bot.should_trade_token(pair_data['baseToken']['address']):
            raise HTTPException(status_code=400, detail="Token failed safety checks")
            
        # Set current pair
        bot.trading_state['current_pair'] = pair_address
        
        return {
            "status": "success",
            "pair_data": pair_data,
            "safety_metrics": bot.trading_state['analyzed_tokens'].get(
                pair_data['baseToken']['address'],
                {}
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle-trading")
async def toggle_trading():
    """Toggle trading with enhanced validation"""
    try:
        if not bot.trading_state['wallet_key']:
            raise HTTPException(status_code=400, detail="Wallet not connected")
            
        if not bot.trading_state['current_pair']:
            raise HTTPException(status_code=400, detail="No token pair selected")
            
        # Additional safety checks before starting trading
        if not bot.trading_state['is_trading']:
            current_token = bot.trading_state['current_pair']
            if not await bot.should_trade_token(current_token):
                raise HTTPException(status_code=400, detail="Token failed safety checks")
        
        bot.trading_state['is_trading'] = not bot.trading_state['is_trading']
        
        if bot.trading_state['is_trading']:
            asyncio.create_task(bot.trading_loop())
            
        return {
            "status": "success",
            "is_trading": bot.trading_state['is_trading'],
            "message": "Trading started" if bot.trading_state['is_trading'] else "Trading stopped"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading-status")
async def get_trading_status():
    """Get comprehensive trading status"""
    return {
        "is_trading": bot.trading_state['is_trading'],
        "current_pair": bot.trading_state['current_pair'],
        "positions": len(bot.trading_state['positions']),
        "daily_stats": bot.risk_manager.daily_stats,
        "performance_metrics": bot.performance_metrics,
        "analyzed_tokens": len(bot.trading_state['analyzed_tokens']),
        "blacklisted_tokens": len(bot.trading_state['blacklisted_tokens'])
    }

@app.get("/token-analysis/{token_address}")
async def get_token_analysis(token_address: str):
    """Get detailed token analysis"""
    analysis = await bot.analyze_token(token_address)
    if not analysis:
        raise HTTPException(status_code=404, detail="Token analysis not found")
    return analysis

@app.get("/performance-metrics")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    return {
        "metrics": bot.performance_metrics,
        "daily_stats": bot.risk_manager.daily_stats,
        "trade_history": bot.trading_state['trade_history'][-50:]  # Last 50 trades
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    bot.trading_state['active_connections'].add(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        if websocket in bot.trading_state['active_connections']:
            bot.trading_state['active_connections'].remove(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize bot on startup"""
    # Initialize price feeds and other necessary components
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    # Close all positions and connections
    if bot.trading_state['positions']:
        await bot.close_all_positions()
    
    # Close all WebSocket connections
    for connection in bot.trading_state['active_connections']:
        await connection.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)