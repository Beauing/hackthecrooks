import asyncio
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import aiohttp

class LiveTradingSimulator:
    def __init__(self, initial_balance: float = 100.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.positions = []
        self.performance_log = []
        self.token_holdings = {}
        self.simulated_wallet = {
            'SOL': initial_balance,  # Start with all funds in SOL
            'tokens': {}  # Will hold other token balances
        }
        
    async def fetch_historical_data(self, pair_address: str, days: int = 7) -> List[Dict]:
    """Fetch real historical data from DexScreener with better error handling"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'pairs' in data and data['pairs']:
                        return data['pairs'][0].get('priceHistory', [])
                    print(f"No data found for pair {pair_address}")
                    return []
                else:
                    print(f"API request failed with status {response.status}")
                    return []
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return []

    async def simulate_trade(self, bot, price_data: Dict, trade_signal: Dict) -> Dict:
        """Simulate trade execution using bot's actual logic"""
        # Apply real-world conditions
        slippage = 0.01  # 1% slippage
        executed_price = price_data['priceUsd'] * (1 + (np.random.uniform(-1, 1) * slippage))
        
        # Calculate trade amount based on bot's position sizing
        trade_amount = self.current_balance * bot.risk_manager.config['max_position_size']
        
        trade_result = {
            'timestamp': datetime.fromtimestamp(price_data['timestamp']),
            'signal': trade_signal['signal'],
            'price': executed_price,
            'amount': trade_amount,
            'balance_before': self.current_balance,
            'confidence': trade_signal['confidence']
        }
        
        if trade_signal['signal'] == "BUY":
            # Check if we have enough SOL
            if self.simulated_wallet['SOL'] >= trade_amount:
                token_amount = trade_amount / executed_price
                self.simulated_wallet['SOL'] -= trade_amount
                self.simulated_wallet['tokens'][price_data['token']] = \
                    self.simulated_wallet['tokens'].get(price_data['token'], 0) + token_amount
                
                self.positions.append({
                    'token': price_data['token'],
                    'size': token_amount,
                    'entry_price': executed_price,
                    'timestamp': datetime.now()
                })
                
        elif trade_signal['signal'] == "SELL":
            # Check if we have tokens to sell
            token = price_data['token']
            if token in self.simulated_wallet['tokens'] and self.simulated_wallet['tokens'][token] > 0:
                token_amount = self.simulated_wallet['tokens'][token]
                sol_received = token_amount * executed_price * (1 - slippage)
                self.simulated_wallet['SOL'] += sol_received
                self.simulated_wallet['tokens'][token] = 0
                
                # Calculate PnL
                matching_position = next((p for p in self.positions if p['token'] == token), None)
                if matching_position:
                    pnl = (executed_price - matching_position['entry_price']) * matching_position['size']
                    self.positions.remove(matching_position)
                    trade_result['pnl'] = pnl
        
        # Update current balance
        self.current_balance = self.simulated_wallet['SOL'] + sum(
            amount * price_data['priceUsd'] 
            for token, amount in self.simulated_wallet['tokens'].items()
        )
        
        trade_result['balance_after'] = self.current_balance
        self.trades.append(trade_result)
        
        return trade_result

    async def run_live_simulation(self, bot, token_pairs: List[str], duration_hours: int = 24):
        """Run simulation using real market data and bot's actual trading logic"""
        start_time = datetime.now()
        print(f"Starting live simulation with ${self.initial_balance}")
        print(f"Monitoring {len(token_pairs)} token pairs")
        print("=" * 50)
        
        while (datetime.now() - start_time).total_seconds() < duration_hours * 3600:
            for pair_address in token_pairs:
                try:
                    # Fetch real market data
                    pair_data = await bot.fetch_dexscreener_data(pair_address)
                    if not pair_data:
                        continue
                        
                    # Run token analysis using bot's actual analysis logic
                    analysis = await bot.analyze_token(pair_data['baseToken']['address'])
                    if not analysis or analysis['safety_score'] < bot.config['min_safety_score']:
                        print(f"Token {pair_address} failed safety checks")
                        continue
                    
                    # Get trading signal using bot's strategy
                    signal = await bot.process_trading_signal({
                        'token_address': pair_data['baseToken']['address'],
                        'price': float(pair_data['priceUsd']),
                        'data': pair_data
                    })
                    
                    # Simulate trade if signal is not HOLD
                    if signal['signal'] != "HOLD":
                        trade_result = await self.simulate_trade(bot, {
                            'timestamp': int(datetime.now().timestamp()),
                            'priceUsd': float(pair_data['priceUsd']),
                            'token': pair_data['baseToken']['address']
                        }, signal)
                        
                        # Log performance
                        self.performance_log.append({
                            'timestamp': datetime.now(),
                            'balance': self.current_balance,
                            'active_positions': len(self.positions),
                            'signal': signal['signal'],
                            'confidence': signal['confidence']
                        })
                        
                        print(f"\nTrade executed for {pair_address}")
                        print(f"Signal: {signal['signal']}")
                        print(f"Current Balance: ${self.current_balance:.2f}")
                        print(f"Active Positions: {len(self.positions)}")
                        
                except Exception as e:
                    print(f"Error processing pair {pair_address}: {str(e)}")
                    
            # Wait before next iteration
            await asyncio.sleep(60)  # Check every minute
            
            # Print periodic updates
            if len(self.performance_log) % 10 == 0:
                hours_elapsed = (datetime.now() - start_time).total_seconds() / 3600
                print(f"\nSimulation Progress: {hours_elapsed:.1f}/{duration_hours} hours")
                print(f"Current Balance: ${self.current_balance:.2f}")
                print(f"Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%")
        
        # Generate final report
        print("\nSimulation Complete")
        print("=" * 50)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%")
        print(f"Number of Trades: {len(self.trades)}")
        print("\nPosition Summary:")
        for token, amount in self.simulated_wallet['tokens'].items():
            if amount > 0:
                print(f"Token {token}: {amount:.6f}")
        print(f"SOL Balance: {self.simulated_wallet['SOL']:.6f}")
        print("=" * 50)
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'trades': self.trades,
            'performance_log': self.performance_log,
            'final_positions': self.simulated_wallet
        }