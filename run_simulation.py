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

class AutomatedSimulator:
    def __init__(self, initial_balance: float = 100.0):
        self.console = Console()
        self.bot = EnhancedTradingBot()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Import API handler methods
        self.fetch_trading_pairs = fetch_trading_pairs.__get__(self)
        self.get_backup_pairs = get_backup_pairs.__get__(self)
        self.filter_promising_pairs = filter_promising_pairs.__get__(self)
        
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

    async def execute_simulated_trade(self, 
                                signal_type: str, 
                                position_size: float, 
                                price: float, 
                                token_address: str) -> Dict:
        """Execute a simulated trade and update portfolio"""
        try:
            self.log_activity(
                f"Executing {signal_type} trade for {position_size:.2f} USD at ${price:.2f}",
                "Trading"
            )

            if signal_type == "BUY":
                if self.simulated_wallet['SOL'] < position_size:
                    self.log_activity("Insufficient balance for trade", "Trading")
                    return {'success': False, 'error': 'Insufficient balance'}

                token_amount = position_size / price
                self.simulated_wallet['SOL'] -= position_size
                self.simulated_wallet['tokens'][token_address] = token_amount
                
                self.log_activity(
                    f"Bought {token_amount:.6f} tokens at ${price:.2f}",
                    "Trading"
                )

            elif signal_type == "SELL":
                if token_address not in self.simulated_wallet['tokens']:
                    self.log_activity("No tokens to sell", "Trading")
                    return {'success': False, 'error': 'No tokens'}

                token_amount = self.simulated_wallet['tokens'][token_address]
                received = token_amount * price
                self.simulated_wallet['SOL'] += received
                del self.simulated_wallet['tokens'][token_address]
                
                self.log_activity(
                    f"Sold {token_amount:.6f} tokens at ${price:.2f}",
                    "Trading"
                )

            # Update current balance
            self.current_balance = self.simulated_wallet['SOL'] + sum(
                amount * price 
                for token, amount in self.simulated_wallet['tokens'].items()
            )

            self.log_activity(f"Updated balance: ${self.current_balance:.2f}", "Portfolio")

            return {
                'success': True,
                'type': signal_type,
                'size': position_size,
                'price': price,
                'token': token_address,
                'balance': self.current_balance
            }

        except Exception as e:
            self.log_activity(f"Trade execution error: {str(e)}", "Error")
            return {'success': False, 'error': str(e)}

    async def process_pair(self, pair):
        """Process a single trading pair with comprehensive logging and validation"""
        try:
            token_address = pair.get('baseToken', {}).get('address')
            if not token_address:
                self.log_activity("Missing base token address in pair", "Error")
                return
            
            self.log_activity(f"Processing pair data: {pair}", "Debug")
            self.log_activity(f"Processing token: {token_address[:8]}...", "Analysis")
            
            # Run token analysis
            self.log_activity("Starting token analysis...", "Analysis")
            analysis = await self.bot.analyze_token(token_address)
            
            if not analysis:
                self.log_activity(f"Analysis failed for {token_address[:8]}...")
                return
            
            self.log_activity(f"Analysis completed with result: {analysis}", "Analysis")
            
            # Check safety score
            if 'safety_score' not in analysis:
                self.log_activity("Missing safety score in analysis", "Error")
                return
                
            if analysis['safety_score'] >= self.bot.config.get('min_safety_score', 0.6):
                # Get trading signal with logging
                self.log_activity("Generating trading signal...", "Trading")
                signal = await self.bot.process_trading_signal({
                    'token_address': token_address,
                    'price': float(pair.get('priceUsd', 0)),
                    'data': pair
                })
                
                self.log_activity(f"Signal received: {signal.get('signal')} with confidence {signal.get('confidence', 0)} {signal.get('reasons')} {signal.get('warnings')}", "Trading")
                
                # Execute trade if we get a signal
                if signal.get('signal') != "HOLD":
                    position_size = self.current_balance * 0.1  # 10% of balance
                    self.log_activity(f"Attempting trade with size: ${position_size:.2f}", "Trading")
                    
                    trade_result = await self.execute_simulated_trade(
                        signal['signal'],
                        position_size,
                        float(pair.get('priceUsd', 0)),
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

    async def run_simulation(self, duration_hours: int = 24):
        """Run simulation with detailed progress tracking"""
        try:
            self.log_activity("Starting simulation", "Startup")
            
            self.bot.trading_state['is_trading'] = True
            self.bot.trading_state['wallet_key'] = 'SIMULATED_WALLET'
            
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            last_pairs_update = start_time - timedelta(hours=1)

            with Live(self.create_status_display(), refresh_per_second=1) as live:
                while datetime.now() < end_time:
                    try:
                        current_time = datetime.now()
                        
                        # Regular status updates
                        if (current_time - self.last_update).seconds >= self.update_interval:
                            time_remaining = end_time - current_time
                            self.log_activity(
                                f"Time remaining: {time_remaining.seconds//3600}h "
                                f"{(time_remaining.seconds//60)%60}m"
                            )
                            self.last_update = current_time
                            live.update(self.create_status_display())

                        if (current_time - last_pairs_update).seconds >= 3600:
                            self.log_activity("Updating trading pairs", "Scanning")
                            pairs = await self.fetch_trading_pairs()
                            
                            if not pairs:
                                self.log_activity("No pairs found, using backup pairs", "Scanning")
                                pairs = self.get_backup_pairs()
                            
                            self.log_activity(f"Found {len(pairs)} pairs before filtering", "Scanning")
                            filtered_pairs = self.filter_promising_pairs(pairs)
                            self.log_activity(f"Filtered down to {len(filtered_pairs)} pairs", "Scanning")
                            
                            last_pairs_update = current_time
                            
                            # Process each pair
                            for pair in filtered_pairs:
                                self.log_activity(f"Starting to process pair: {pair}", "Debug")
                                await self.process_pair(pair)
                                live.update(self.create_status_display())
                            
                    except Exception as e:
                        self.log_activity(f"Error in main loop: {str(e)}", "Error")
                        live.update(self.create_status_display())
                    
                    await asyncio.sleep(1)
                    
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