import asyncio
import aiohttp
import json
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.transaction import Transaction
from solana.rpc.commitment import Commitment
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import base58
from cryptography.fernet import Fernet
import os
import numpy as np
from decimal import Decimal
import requests

class TradingStrategy:
    def __init__(self, short_window=20, long_window=50, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def calculate_moving_averages(self, prices):
        prices_array = np.array(prices)
        short_ma = np.mean(prices_array[-self.short_window:])
        long_ma = np.mean(prices_array[-self.long_window:])
        return short_ma, long_ma

    def calculate_rsi(self, prices):
        if len(prices) < self.rsi_period + 1:
            return 50  # Default neutral value
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_signal(self, prices):
        if len(prices) < self.long_window:
            return "WAIT"
        
        short_ma, long_ma = self.calculate_moving_averages(prices)
        rsi = self.calculate_rsi(prices)
        
        # Combined signal based on RSI and MA crossover
        if short_ma > long_ma and rsi < self.rsi_oversold:
            return "BUY"
        elif short_ma < long_ma and rsi > self.rsi_overbought:
            return "SELL"
        return "HOLD"

class WalletManager:
    def __init__(self):
        self.key = None
        self.fernet = None
        self.encrypted_private_key = None
        
    def setup_encryption(self):
        if not os.path.exists('key.key'):
            self.key = Fernet.generate_key()
            with open('key.key', 'wb') as key_file:
                key_file.write(self.key)
        else:
            with open('key.key', 'rb') as key_file:
                self.key = key_file.read()
        self.fernet = Fernet(self.key)

    def store_private_key(self, private_key):
        if not self.fernet:
            self.setup_encryption()
        self.encrypted_private_key = self.fernet.encrypt(private_key.encode())
        
    def get_private_key(self):
        if self.encrypted_private_key:
            return self.fernet.decrypt(self.encrypted_private_key).decode()
        return None

class SolanaTradeBot:
    def __init__(self):
        # Initialize Solana client
        self.client = AsyncClient("https://api.mainnet-beta.solana.com")
        self.wallet_manager = WalletManager()
        
        # Trading parameters
        self.max_slippage = 0.01  # 1%
        self.min_sol_balance = 0.05  # Minimum SOL to keep for fees
        self.trading_strategy = TradingStrategy()
        
        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Solana Trading Bot")
        self.setup_gui()
        
        # Trading state
        self.is_trading = False
        self.chart_data = []
        self.current_token_pair = None
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Token pair selection
        pair_frame = ttk.LabelFrame(main_frame, text="Token Pair", padding="5")
        pair_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(pair_frame, text="Token Pair Address:").grid(row=0, column=0, padx=5)
        self.pair_entry = ttk.Entry(pair_frame, width=50)
        self.pair_entry.grid(row=0, column=1, padx=5)
        ttk.Button(pair_frame, text="Load Pair", 
                  command=self.load_token_pair).grid(row=0, column=2, padx=5)
        
        # Trading controls
        controls_frame = ttk.LabelFrame(main_frame, text="Trading Controls", padding="5")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(controls_frame, text="Connect Wallet", 
                  command=self.connect_wallet).grid(row=0, column=0, padx=5)
        self.trade_button = ttk.Button(controls_frame, text="Start Trading", 
                                     command=self.toggle_trading)
        self.trade_button.grid(row=0, column=1, padx=5)
        
        # Strategy parameters
        strategy_frame = ttk.LabelFrame(main_frame, text="Strategy Parameters", padding="5")
        strategy_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(strategy_frame, text="Max Slippage (%):").grid(row=0, column=0, padx=5)
        self.slippage_entry = ttk.Entry(strategy_frame, width=10)
        self.slippage_entry.insert(0, "1.0")
        self.slippage_entry.grid(row=0, column=1, padx=5)
        
        # Chart frame
        self.chart_frame = ttk.LabelFrame(main_frame, text="Price Chart", padding="5")
        self.chart_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Not connected")
        self.status_label.grid(row=0, column=0, padx=5)

    async def get_jupiter_quote(self, input_mint, output_mint, amount, slippage):
        """Get quote from Jupiter aggregator"""
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
                return None

    async def execute_jupiter_swap(self, quote):
        """Execute swap on Jupiter"""
        # Get user wallet
        private_key = self.wallet_manager.get_private_key()
        if not private_key:
            raise Exception("Wallet not connected")
            
        # Create transaction
        url = "https://quote-api.jup.ag/v4/swap"
        data = {
            "quoteResponse": quote,
            "userPublicKey": base58.b58encode(Keypair.from_secret_key(
                base58.b58decode(private_key)).public_key).decode(),
            "wrapUnwrapSOL": True
        }
        
        async with aiohttp.ClientSession() as session:
            # Get swap transaction
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    raise Exception("Failed to create swap transaction")
                swap_data = await response.json()
                
            # Sign and send transaction
            transaction = Transaction.deserialize(swap_data['swapTransaction'])
            signed_tx = transaction.sign(Keypair.from_secret_key(
                base58.b58decode(private_key)))
            
            # Submit transaction
            result = await self.client.send_transaction(signed_tx)
            return result

    def load_token_pair(self):
        """Load and validate token pair"""
        pair_address = self.pair_entry.get().strip()
        if not pair_address:
            messagebox.showerror("Error", "Please enter a token pair address")
            return
            
        self.current_token_pair = pair_address
        self.status_label.config(text=f"Loaded pair: {pair_address[:10]}...")

    async def fetch_dexscreener_data(self, pair_address):
        """Fetch price data from DexScreener API"""
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'pairs' in data and len(data['pairs']) > 0:
                        return data['pairs'][0]
                return None

    async def update_chart(self):
        """Update the price chart with new data"""
        self.ax.clear()
        
        prices = [float(point['priceUsd']) for point in self.chart_data]
        timestamps = [datetime.fromtimestamp(point['timestamp']) for point in self.chart_data]
        
        # Plot price
        self.ax.plot(timestamps, prices, label='Price')
        
        # Plot moving averages if enough data
        if len(prices) >= self.trading_strategy.long_window:
            short_ma, long_ma = self.trading_strategy.calculate_moving_averages(prices)
            self.ax.axhline(y=short_ma, color='r', linestyle='--', label='Short MA')
            self.ax.axhline(y=long_ma, color='g', linestyle='--', label='Long MA')
        
        self.ax.set_title("Token Price (USD)")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        
        plt.xticks(rotation=45)
        self.canvas.draw()

    async def execute_trade(self, side: str, amount: float, price: float):
        """Execute a trade on Jupiter Exchange"""
        try:
            if not self.current_token_pair:
                raise Exception("No token pair selected")
                
            # Get token addresses from DexScreener data
            pair_data = await self.fetch_dexscreener_data(self.current_token_pair)
            if not pair_data:
                raise Exception("Failed to fetch pair data")
                
            input_mint = pair_data['baseToken']['address'] if side == "sell" else "So11111111111111111111111111111111111111112"
            output_mint = "So11111111111111111111111111111111111111112" if side == "sell" else pair_data['baseToken']['address']
            
            # Calculate amount in lamports/smallest unit
            amount_in = int(amount * (10 ** pair_data['baseToken']['decimals']))
            
            # Get quote
            quote = await self.get_jupiter_quote(input_mint, output_mint, amount_in, self.max_slippage)
            if not quote:
                raise Exception("Failed to get quote")
                
            # Execute swap
            result = await self.execute_jupiter_swap(quote)
            self.status_label.config(text=f"Trade executed: {side} {amount} at ${price}")
            
            return result
            
        except Exception as e:
            error_msg = f"Trade execution error: {str(e)}"
            self.status_label.config(text=error_msg)
            messagebox.showerror("Trade Error", error_msg)
            return None

    def connect_wallet(self):
        """Connect to Phantom wallet"""
        # In a production environment, you would use Phantom's wallet adapter
        # For this example, we'll use a simple dialog
        private_key = tk.simpledialog.askstring("Connect Wallet", 
                                              "Enter your private key:",
                                              show='*')
        if private_key:
            try:
                # Validate private key
                keypair = Keypair.from_secret_key(base58.b58decode(private_key))
                self.wallet_manager.store_private_key(private_key)
                self.status_label.config(text=f"Connected: {str(keypair.public_key)[:10]}...")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid private key: {str(e)}")

    def toggle_trading(self):
        """Toggle automated trading on/off"""
        self.is_trading = not self.is_trading
        if self.is_trading:
            if not self.wallet_manager.get_private_key():
                messagebox.showerror("Error", "Please connect wallet first")
                self.is_trading = False
                return
                
            if not self.current_token_pair:
                messagebox.showerror("Error", "Please load a token pair first")
                self.is_trading = False
                return
                
            self.trade_button.config(text="Stop Trading")
            self.status_label.config(text="Trading started")
            asyncio.create_task(self.trading_loop())
        else:
            self.trade_button.config(text="Start Trading")
            self.status_label.config(text="Trading stopped")

    async def trading_loop(self):
        """Main trading loop"""
        while self.is_trading:
            try:
                # Fetch latest price data
                pair_data = await self.fetch_dexscreener_data(self.current_token_pair)
                if pair_data:
                    current_price = float(pair_data['priceUsd'])
                    self.chart_data.append({
                        'timestamp': int(datetime.now().timestamp()),
                        'priceUsd': current_price
                    })
                    # Keep only last 24 hours of data
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.chart_data = [point for point in self.chart_data 
                                     if datetime.fromtimestamp(point['timestamp']) > cutoff]
                    
                    # Update chart
                    await self.update_chart()
                    
                    # Get trading signal
                    prices = [float(point['priceUsd']) for point in self.chart_data]
                    signal = self.trading_strategy.get_signal(prices)
                    
                    # Check wallet balance
                    wallet_pubkey = Keypair.from_secret_key(
                        base58.b58decode(self.wallet_manager.get_private_key())
                    ).public_key
                    balance = await self.client.get_balance(wallet_pubkey)
                    sol_balance = float(balance.value) / 1e9  # Convert lamports to SOL
                    
                    # Execute trades based on signal
                    if signal == "BUY" and sol_balance > self.min_sol_balance:
                        trade_amount = (sol_balance - self.min_sol_balance) * 0.5  # Use 50% of available balance
                        await self.execute_trade("buy", trade_amount, current_price)
                        self.status_label.config(text=f"BUY signal executed at ${current_price}")
                    
                    elif signal == "SELL":
                        # Get token balance (this is simplified, you'd need to get actual token balance)
                        token_balance = float(pair_data['liquidity'])  # This should be replaced with actual balance check
                        if token_balance > 0:
                            await self.execute_trade("sell", token_balance * 0.5, current_price)  # Sell 50% of holdings
                            self.status_label.config(text=f"SELL signal executed at ${current_price}")
                    
                    else:
                        self.status_label.config(text=f"HOLD signal at ${current_price}")
                
            except Exception as e:
                error_msg = f"Error in trading loop: {str(e)}"
                self.status_label.config(text=error_msg)
                print(error_msg)  # For debugging
            
            await asyncio.sleep(60)  # Wait 1 minute before next update

    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    # Set up async event loop
    bot = SolanaTradeBot()
    
    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the application
    try:
        bot.run()
    finally:
        loop.close()