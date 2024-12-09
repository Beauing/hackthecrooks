import asyncio
from datetime import datetime, timedelta
from rich.console import Console
from rich.live import Live
from rich.table import Table
from main import EnhancedTradingBot
from trading.api_handler import fetch_trading_pairs, get_backup_pairs, filter_promising_pairs
import aiohttp
import time
import json
from typing import Dict, List, Optional

class OrderType:
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"

class LimitOrder:
    def __init__(self, 
                 order_type: str,
                 token_address: str,
                 amount: float,
                 limit_price: float,
                 stop_loss: float = None,
                 take_profit: float = None):
        self.order_type = order_type
        self.token_address = token_address
        self.amount = amount
        self.limit_price = limit_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.created_at = datetime.now()
        self.filled = False
        self.cancelled = False

class AutomatedSimulator:
    def __init__(self, initial_balance: float = 100.0):
        self.console = Console()
        self.bot = EnhancedTradingBot()
        self.initial_balance = initial_balance
        # Import API handler methods
        self.fetch_trading_pairs = fetch_trading_pairs.__get__(self)
        self.get_backup_pairs = get_backup_pairs.__get__(self)
        self.filter_promising_pairs = filter_promising_pairs.__get__(self)
        self.current_balance = initial_balance
        
        # Initialize trading state dictionary
        self.trading_state = {
            'is_trading': False,
            'current_pair': None,
            'chart_data': [],
            'wallet_key': None,
            'positions': [],
            'active_connections': set(),
            'analyzed_tokens': {},
            'pending_orders': [],  # For limit orders
            'blacklisted_tokens': set()
        }
        
        # Initialize simulated wallet
        self.simulated_wallet = {
            'SOL': initial_balance,
            'tokens': {}
        }
        
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        self.last_update = datetime.now()
        self.update_interval = 15
        self.active_pairs = set()
        self.recent_updates = []
        self.current_stage = "Initializing"

        # Add new limit order tracking
        self.orders = {
            'pending': [],  # For limit orders waiting to be filled
            'filled': [],   # For completed orders
            'cancelled': [] # For cancelled orders
        }

    def log_activity(self, message: str, stage: str = None):
        """Log activity with timestamp and update current stage"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if stage:
            self.current_stage = stage
        
        log_entry = {
            'time': datetime.now(),
            'message': message,
            'stage': self.current_stage
        }
        self.recent_updates.append(log_entry)
        
        # Keep only last 50 updates
        if len(self.recent_updates) > 50:
            self.recent_updates = self.recent_updates[-50:]
        
        # Print to console
        self.console.print(f"[cyan]{timestamp}[/cyan] | [yellow]{self.current_stage}[/yellow] | {message}")
    
    async def monitor_orders(self, current_price: float):
        """Monitor pending limit orders"""
        for order in self.trading_state['pending_orders'][:]:  # Use slice copy to modify list while iterating
            if order.cancelled:
                continue
                
            if order.order_type == OrderType.LIMIT_BUY:
                # Check if price dropped to limit price
                if current_price <= order.limit_price:
                    # Execute the buy
                    token_amount = order.amount / order.limit_price
                    self.simulated_wallet['SOL'] -= order.amount
                    self.simulated_wallet['tokens'][order.token_address] = token_amount
                    
                    order.filled = True
                    self.log_activity(
                        f"Limit buy order filled: {token_amount:.6f} tokens at ${order.limit_price:.2f}",
                        "Trading"
                    )
                    
            elif order.order_type == OrderType.LIMIT_SELL:
                # Check if price rose to limit price
                if current_price >= order.limit_price:
                    # Execute the sell
                    if order.token_address in self.simulated_wallet['tokens']:
                        token_amount = self.simulated_wallet['tokens'][order.token_address]
                        received = token_amount * order.limit_price
                        self.simulated_wallet['SOL'] += received
                        del self.simulated_wallet['tokens'][order.token_address]
                        
                        order.filled = True
                        self.log_activity(
                            f"Limit sell order filled: {token_amount:.6f} tokens at ${order.limit_price:.2f}",
                            "Trading"
                        )

            # Cancel orders older than 24 hours
            if (datetime.now() - order.created_at).total_seconds() > 24 * 3600:
                order.cancelled = True
                self.log_activity(f"Order cancelled due to timeout", "Trading")

            # Remove filled or cancelled orders
            if order.filled or order.cancelled:
                self.trading_state['pending_orders'].remove(order)

        # Update current balance
        self.current_balance = self.simulated_wallet['SOL'] + sum(
            amount * current_price 
            for token, amount in self.simulated_wallet['tokens'].items()
        )

    def create_status_display(self):
        """Create enhanced status display"""
        status_table = Table(title=f"Trading Bot Simulation - {self.current_stage}")
        status_table.add_column("Metric", style="cyan", width=20)
        status_table.add_column("Value", style="green", width=30)
        
        # Financial metrics
        status_table.add_row(
            "Current Balance", 
            f"${self.current_balance:.2f}"
        )
        
        pnl = self.current_balance - self.initial_balance
        pnl_style = "green" if pnl >= 0 else "red"
        status_table.add_row(
            "P&L", 
            f"${pnl:.2f}",
            style=pnl_style
        )
        
        status_table.add_row("Active Pairs", str(len(self.active_pairs)))
        status_table.add_row("Total Trades", str(self.performance_metrics['total_trades']))
        
        # Win rate if we have trades
        if self.performance_metrics['total_trades'] > 0:
            win_rate = (self.performance_metrics['winning_trades'] / 
                       self.performance_metrics['total_trades'] * 100)
            status_table.add_row("Win Rate", f"{win_rate:.1f}%")
        
        # Add recent activity section
        if self.recent_updates:
            status_table.add_section()
            status_table.add_row("Recent Activities", style="bold yellow")
            
            # Show last 5 updates in reverse order (newest first)
            for update in reversed(self.recent_updates[-5:]):
                time_str = update['time'].strftime("%H:%M:%S")
                status_table.add_row(
                    time_str,
                    f"{update['stage']}: {update['message']}",
                    style="dim"
                )
        
        return status_table

    async def track_simulation_state(self, current_price: float = None):
        """Track and log all simulation states"""
        try:
            # Create state snapshot
            state_snapshot = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'balance': self.current_balance,
                'pending_orders': len(self.trading_state['pending_orders']),
                'active_positions': len(self.trading_state['positions']),
                'current_price': current_price,
            }

            # Track pending orders
            if self.trading_state['pending_orders']:
                self.log_activity(
                    f"Pending Orders: {len(self.trading_state['pending_orders'])} - "
                    f"Buy: {sum(1 for o in self.trading_state['pending_orders'] if o.order_type == OrderType.LIMIT_BUY)} "
                    f"Sell: {sum(1 for o in self.trading_state['pending_orders'] if o.order_type == OrderType.LIMIT_SELL)}",
                    "State"
                )

            # Track positions
            if self.trading_state['positions']:
                positions_value = sum(
                    pos.size * current_price if current_price else pos.size * pos.entry_price
                    for pos in self.trading_state['positions']
                )
                self.log_activity(
                    f"Open Positions Value: ${positions_value:.2f}",
                    "State"
                )

            # Track wallet contents
            token_holdings = []
            for token, amount in self.simulated_wallet['tokens'].items():
                token_holdings.append(f"{amount:.6f} of {token[:8]}")
            
            if token_holdings:
                self.log_activity(
                    f"Token Holdings: {', '.join(token_holdings)}",
                    "State"
                )

            # Track performance
            if self.performance_metrics['total_trades'] > 0:
                win_rate = (self.performance_metrics['winning_trades'] / 
                        self.performance_metrics['total_trades'] * 100)
                self.log_activity(
                    f"Performance - Win Rate: {win_rate:.1f}% "
                    f"Best Trade: ${self.performance_metrics['best_trade']:.2f} "
                    f"Total P/L: ${self.performance_metrics['total_pnl']:.2f}",
                    "State"
                )

            return state_snapshot

        except Exception as e:
            self.log_activity(f"Error tracking state: {str(e)}", "Error")
            return None
    async def execute_simulated_trade(self, 
                                    signal_type: str, 
                                    position_size: float, 
                                    current_price: float, 
                                    token_address: str) -> Dict:
        """Execute a simulated limit order trade"""
        try:
            # Calculate limit price with 1% better than market
            if signal_type == "BUY":
                limit_price = current_price * 0.99  # Buy 1% lower
                self.log_activity(
                    f"Creating limit buy order for {position_size:.2f} USD at ${limit_price:.2f}",
                    "Trading"
                )
                
                # Create limit buy order
                order = LimitOrder(
                    OrderType.LIMIT_BUY,
                    token_address,
                    position_size,
                    limit_price,
                    stop_loss=limit_price * 0.95,  # 5% stop loss
                    take_profit=limit_price * 1.05  # 5% take profit
                )
                
            else:  # SELL
                limit_price = current_price * 1.01  # Sell 1% higher
                self.log_activity(
                    f"Creating limit sell order for {position_size:.2f} USD at ${limit_price:.2f}",
                    "Trading"
                )
                
                # Create limit sell order
                order = LimitOrder(
                    OrderType.LIMIT_SELL,
                    token_address,
                    position_size,
                    limit_price,
                    stop_loss=limit_price * 0.95,
                    take_profit=limit_price * 1.05
                )

            # Add order to pending orders
            self.orders['pending'].append(order)
            self.log_activity(f"Limit order created at ${limit_price:.2f}", "Trading")

            return {
                'success': True,
                'type': signal_type,
                'size': position_size,
                'limit_price': limit_price,
                'token': token_address
            }

        except Exception as e:
            self.log_activity(f"Order creation error: {str(e)}", "Error")
            return {'success': False, 'error': str(e)}

    async def load_and_track_price_data(self, pair):
        """Load and track price data using working search API"""
        try:
            pair_address = pair.get('pairAddress')
            self.log_activity("Loading price data...", "Data")
            
            async with aiohttp.ClientSession() as session:
                # Use the working search API endpoint
                url = f"https://api.dexscreener.com/latest/dex/search?q={pair_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'pairs' in data and data['pairs']:
                            pair_data = data['pairs'][0]
                            
                            # Initialize timeframes if needed
                            if 'price_data' not in self.bot.trading_state:
                                self.bot.trading_state['price_data'] = {
                                    '1m': [],
                                    '5m': [],
                                    '15m': []
                                }
                            
                            # Create price point
                            price_point = {
                                'timestamp': int(datetime.now().timestamp()),
                                'price': float(pair_data.get('priceUsd', 0)),
                                'volume': float(pair_data.get('volume', {}).get('h24', 0)),
                                'high': float(pair_data.get('priceMax24h', pair_data.get('priceUsd', 0))),
                                'low': float(pair_data.get('priceMin24h', pair_data.get('priceUsd', 0)))
                            }
                            
                            # Update price data for all timeframes immediately
                            await self.update_timeframes(price_point)
                            
                            # Check data status
                            data_status = self.get_data_completeness()
                            self.display_data_progress(data_status)
                            
                            return data_status['ready_for_trading']
                    
            self.log_activity("Failed to fetch price data", "Error")
            return False

        except Exception as e:
            self.log_activity(f"Error loading price data: {str(e)}", "Error")
            return False

    async def update_timeframes(self, price_point):
        """Update timeframe data on each price update"""
        now = datetime.fromtimestamp(price_point['timestamp'])
        
        # Always update 1m data
        self.bot.trading_state['price_data']['1m'].append(price_point)
        self.log_activity(f"Added 1m data point, total: {len(self.bot.trading_state['price_data']['1m'])}", "Data")
        
        # Update 5m data if new period
        if len(self.bot.trading_state['price_data']['5m']) == 0 or \
                now.minute % 5 == 0 and self.is_new_period(now, self.bot.trading_state['price_data']['5m'][-1]['timestamp'], 300):
            self.bot.trading_state['price_data']['5m'].append(price_point)
            self.log_activity(f"Added 5m candle, total: {len(self.bot.trading_state['price_data']['5m'])}", "Data")
        
        # Update 15m data if new period
        if len(self.bot.trading_state['price_data']['15m']) == 0 or \
                now.minute % 15 == 0 and self.is_new_period(now, self.bot.trading_state['price_data']['15m'][-1]['timestamp'], 900):
            self.bot.trading_state['price_data']['15m'].append(price_point)
            self.log_activity(f"Added 15m candle, total: {len(self.bot.trading_state['price_data']['15m'])}", "Data")

        # Keep within limits
        max_points = {'1m': 60, '5m': 24, '15m': 16}
        for tf, limit in max_points.items():
            if len(self.bot.trading_state['price_data'][tf]) > limit:
                self.bot.trading_state['price_data'][tf] = self.bot.trading_state['price_data'][tf][-limit:]

    def is_new_period(self, current_time: datetime, last_timestamp: int, period_seconds: int) -> bool:
        """Check if we're in a new time period"""
        last_time = datetime.fromtimestamp(last_timestamp)
        return (current_time - last_time).seconds >= period_seconds

    def get_data_completeness(self) -> Dict:
        """Calculate data completeness"""
        min_required = {
            '1m': 30,
            '5m': 24,
            '15m': 16
        }
        
        completeness = {}
        for timeframe, required in min_required.items():
            current = len(self.bot.trading_state['price_data'][timeframe])
            completeness[timeframe] = {
                'current': current,
                'required': required,
                'percentage': (current / required) * 100 if current < required else 100
            }

        return {
            'completeness': completeness,
            'ready_for_trading': all(
                len(data) >= min_required[tf] 
                for tf, data in self.bot.trading_state['price_data'].items()
            )
        }

    def display_data_progress(self, data_status):
        """Display data collection progress"""
        progress_message = "Data Collection Progress:\n"
        for timeframe, status in data_status['completeness'].items():
            bars = '█' * int(status['percentage'] // 10) + '░' * (10 - int(status['percentage'] // 10))
            progress_message += f"{timeframe}: {bars} {status['current']}/{status['required']} ({status['percentage']:.1f}%)\n"
        
        self.log_activity(progress_message, "Progress")

    def convert_and_store_historical_data(self, historical_data):
        """Convert and store historical data"""
        try:
            for candle in historical_data:
                price_point = {
                    'timestamp': int(candle['timestamp']),
                    'price': float(candle['close']),
                    'volume': float(candle['volume']),
                    'high': float(candle['high']),
                    'low': float(candle['low'])
                }
                self.update_timeframes(price_point)
        except Exception as e:
            self.log_activity(f"Error converting historical data: {str(e)}", "Error")


    async def process_pair(self, pair):
        """Process trading pair with comprehensive data tracking"""
        try:
            token_address = pair.get('baseToken', {}).get('address')
            if not token_address:
                self.log_activity("Missing base token address in pair", "Error")
                return

            # Load and track price data across timeframes
            ready_for_trading = await self.load_and_track_price_data(pair)
            
            if not ready_for_trading:
                return
                
            current_price = float(pair.get('priceUsd', 0))
            
            # Add price point to main chart data
            price_point = {
                'timestamp': int(datetime.now().timestamp()),
                'price': current_price,
                'volume': float(pair.get('volume', {}).get('h24', 0)),
                'high': float(pair.get('priceMax24h', current_price)),
                'low': float(pair.get('priceMin24h', current_price))
            }

            # Initialize and update chart data
            if 'chart_data' not in self.bot.trading_state:
                self.bot.trading_state['chart_data'] = []
                
            self.bot.trading_state['chart_data'].append(price_point)

            # Log data point addition
            self.log_activity(
                f"Added price point: ${current_price}. Total points: {len(self.bot.trading_state['chart_data'])}",
                "Debug"
            )

            # Track time between points
            if len(self.bot.trading_state['chart_data']) >= 2:
                last_time = self.bot.trading_state['chart_data'][-1]['timestamp']
                prev_time = self.bot.trading_state['chart_data'][-2]['timestamp']
                time_diff = last_time - prev_time
                self.log_activity(f"Time between points: {time_diff} seconds", "Debug")

            # Maintain rolling window of 100 points
            if len(self.bot.trading_state['chart_data']) > 100:
                self.bot.trading_state['chart_data'] = self.bot.trading_state['chart_data'][-100:]

            # Check if we have enough data points
            current_points = len(self.bot.trading_state['chart_data'])
            required_points = 50  # For EMA calculation
            
            self.log_activity(
                f"Price history: {current_points}/{required_points} points needed for trading",
                "Analysis"
            )

            # Wait for more data if needed
            if current_points < required_points:
                self.log_activity(
                    f"Need {required_points - current_points} more points before trading",
                    "Analysis"
                )
                return

            # Proceed with analysis
            self.log_activity(f"Processing token: {token_address[:8]}...", "Analysis")
            analysis = await self.bot.analyze_token(token_address)
            
            if not analysis:
                self.log_activity(f"Analysis failed for {token_address[:8]}...")
                return
                
            self.log_activity(f"Analysis completed with result: {analysis}", "Analysis")
            
            # Verify safety score
            if 'safety_score' not in analysis:
                self.log_activity("Missing safety score in analysis", "Error")
                return
                
            if analysis['safety_score'] >= self.bot.config.get('min_safety_score', 0.6):
                # Generate trading signal
                self.log_activity("Generating trading signal...", "Trading")
                signal = await self.bot.process_trading_signal({
                    'token_address': token_address,
                    'price': current_price,
                    'data': pair
                })
                
                self.log_activity(
                    f"Signal received: {signal.get('signal')} with confidence {signal.get('confidence', 0)} "
                    f"{signal.get('reasons')} {signal.get('warnings')}", 
                    "Trading"
                )
                
                # Execute trade for non-HOLD signals
                if signal.get('signal') != "HOLD":
                    position_size = self.current_balance * 0.1
                    self.log_activity(f"Attempting trade with size: ${position_size:.2f}", "Trading")
                    
                    trade_result = await self.execute_simulated_trade(
                        signal['signal'],
                        position_size,
                        current_price,
                        token_address
                    )
                    
                    if trade_result.get('success'):
                        self.log_activity(
                            f"Trade successful - New balance: ${self.current_balance:.2f}",
                            "Trading"
                        )
                    else:
                        self.log_activity(
                            f"Trade failed: {trade_result.get('error', 'Unknown error')}",
                            "Error"
                        )
            else:
                self.log_activity(f"Token failed safety check. Score: {analysis['safety_score']}", "Analysis")
                        
        except Exception as e:
            import traceback
            self.log_activity(f"Error processing pair: {str(e)}", "Error")
            self.log_activity(f"Traceback: {traceback.format_exc()}", "Error")


    def aggregate_candle(self, candles: List[Dict]) -> Dict:
        """Aggregate multiple candles into a single candle"""
        if not candles:
            return None
            
        return {
            'timestamp': candles[-1]['timestamp'],
            'price': candles[-1]['price'],
            'high': max(c['high'] for c in candles),
            'low': min(c['low'] for c in candles),
            'volume': sum(c['volume'] for c in candles)
        }

    async def execute_simulated_trade(self, signal, opportunity, safety_score):
        """Execute and track simulated trade"""
        try:
            self.log_activity(
                f"Executing {signal['signal']} for {opportunity['token_address'][:8]}...", 
                "Trading"
            )
            
            # Calculate position size
            position_size = self.current_balance * self.bot.risk_manager.config['max_position_size']
            position_size *= safety_score
            
            # Apply simulated slippage
            slippage = 0.01  # 1% slippage
            executed_price = float(opportunity['price']) * (1 + (signal['signal'] == "BUY" and slippage or -slippage))
            
            trade = {
                'timestamp': datetime.now(),
                'token': opportunity['token_address'],
                'signal': signal['signal'],
                'price': executed_price,
                'size': position_size,
                'safety_score': safety_score
            }
            
            if signal['signal'] == "BUY":
                # Execute buy
                token_amount = position_size / executed_price
                self.simulated_wallet['SOL'] -= position_size
                self.simulated_wallet['tokens'][opportunity['token_address']] = token_amount
                self.log_activity(f"Bought {token_amount:.6f} tokens at ${executed_price:.4f}")
                
            else:  # SELL
                # Execute sell
                if opportunity['token_address'] in self.simulated_wallet['tokens']:
                    token_amount = self.simulated_wallet['tokens'][opportunity['token_address']]
                    sol_received = token_amount * executed_price * (1 - slippage)
                    trade['pnl'] = sol_received - position_size
                    
                    self.simulated_wallet['SOL'] += sol_received
                    del self.simulated_wallet['tokens'][opportunity['token_address']]
                    
                    # Update metrics
                    self.performance_metrics['total_trades'] += 1
                    if trade['pnl'] > 0:
                        self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_pnl'] += trade['pnl']
                    self.performance_metrics['best_trade'] = max(
                        self.performance_metrics['best_trade'],
                        trade['pnl']
                    )
                    self.performance_metrics['worst_trade'] = min(
                        self.performance_metrics['worst_trade'],
                        trade['pnl']
                    )
                    
                    self.log_activity(
                        f"Sold {token_amount:.6f} tokens at ${executed_price:.4f}, "
                        f"PnL: ${trade['pnl']:.2f}"
                    )
            
            # Record trade and update balance
            self.trades.append(trade)
            self.current_balance = self.simulated_wallet['SOL'] + sum(
                amount * float(opportunity['price'])
                for token, amount in self.simulated_wallet['tokens'].items()
            )
            
        except Exception as e:
            self.log_activity(f"Trade execution error: {str(e)}", "Error")

    async def trading_loop(self, end_time: datetime, last_pairs_update: datetime, live_display):
        last_data_update = datetime.now()
        tracked_pairs = []
        self.bot.trading_state['chart_data'] = []  # Initialize empty chart data

        while datetime.now() < end_time:
            try:
                current_time = datetime.now()
                
                if (current_time - self.last_update).seconds >= self.update_interval:
                    time_remaining = end_time - current_time
                    self.log_activity(f"Time remaining: {time_remaining.seconds//3600}h {(time_remaining.seconds//60)%60}m")
                    
                    # Safe access to current price
                    current_price = None
                    if self.bot.trading_state['chart_data']:
                        current_price = self.bot.trading_state['chart_data'][-1].get('price')
                    
                    if current_price:
                        await self.track_simulation_state(current_price)
                    self.last_update = current_time
                    live_display.update(self.create_status_display())

                # Discover pairs every 5 minutes
                if (current_time - last_pairs_update).seconds >= 300:
                    self.log_activity("Updating trading pairs list", "Scanning")
                    pairs = await self.fetch_trading_pairs()
                    if not pairs:
                        pairs = self.get_backup_pairs()
                    filtered_pairs = self.filter_promising_pairs(pairs)
                    tracked_pairs = filtered_pairs
                    last_pairs_update = current_time
                    self.log_activity(f"Now tracking {len(tracked_pairs)} pairs", "Scanning")

                # Update price data every minute
                if (current_time - last_data_update).seconds >= 60:
                    for pair in tracked_pairs:
                        await self.load_and_track_price_data(pair)
                        if self.ready_for_trading(pair):
                            await self.process_pair(pair)
                    last_data_update = current_time

                # Monitor existing orders
                if self.trading_state.get('pending_orders'):
                    current_price = self.trading_state.get('chart_data', [{}])[-1].get('price')
                    await self.monitor_orders(current_price)

                await asyncio.sleep(1)
                
            except Exception as e:
                self.log_activity(f"Error in trading loop: {str(e)}", "Error")
                live_display.update(self.create_status_display())

    def ready_for_trading(self, pair) -> bool:
        """Check if we have enough data to trade"""
        return all(
            len(self.bot.trading_state['price_data'][tf]) >= min_points
            for tf, min_points in {'1m': 30, '5m': 24, '15m': 16}.items()
        )

    async def run_simulation(self, duration_hours: int = 24):
        """Run simulation with detailed progress tracking"""
        try:
            self.log_activity("Starting simulation", "Startup")
            
            # Initialize simulation state
            self.bot.trading_state['is_trading'] = True
            self.bot.trading_state['wallet_key'] = 'SIMULATED_WALLET'
            self.trading_state['pending_orders'] = []  # Add this for limit orders
            
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            last_pairs_update = start_time - timedelta(hours=1)

            with Live(self.create_status_display(), refresh_per_second=1) as live:
                await self.trading_loop(end_time, last_pairs_update, live)
                
            self.log_activity("Simulation completed", "Complete")
            self.print_final_results()
            
        except Exception as e:
            self.log_activity(f"Critical simulation error: {str(e)}", "Fatal Error")
            raise

    def print_final_results(self):
        """Print detailed final results"""
        table = Table(title="Simulation Final Results")
        table.add_column("Metric")
        table.add_column("Value")
        
        # Add results rows
        table.add_row("Initial Balance", f"${self.initial_balance:.2f}")
        table.add_row("Final Balance", f"${self.current_balance:.2f}")
        table.add_row(
            "Total Return",
            f"{((self.current_balance/self.initial_balance - 1) * 100):.2f}%"
        )
        table.add_row("Total Trades", str(self.performance_metrics['total_trades']))
        
        if self.performance_metrics['total_trades'] > 0:
            table.add_row(
                "Win Rate",
                f"{(self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades'] * 100):.1f}%"
            )
        
        table.add_row("Best Trade", f"${self.performance_metrics['best_trade']:.2f}")
        table.add_row("Worst Trade", f"${self.performance_metrics['worst_trade']:.2f}")
        
        self.console.print(table)

async def main():
    try:
        simulator = AutomatedSimulator(initial_balance=100.0)
        await simulator.run_simulation(duration_hours=24)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())