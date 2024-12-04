import asyncio
from datetime import datetime
from trading.live_simulator import LiveTradingSimulator
from main import EnhancedTradingBot  # Add this import
from rich.console import Console
from rich.table import Table
from rich.live import Live

class SimulationRunner:
    def __init__(self, initial_balance: float = 100.0):
        self.console = Console()
        self.bot = EnhancedTradingBot()
        self.simulator = LiveTradingSimulator(initial_balance)
        self.running_pl = 0
        self.trade_count = 0
        
    def create_status_table(self):
        """Create rich table for live status display"""
        table = Table(title="Trading Bot Simulation Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Initial Balance", f"${self.simulator.initial_balance:.2f}")
        table.add_row("Current Balance", f"${self.simulator.current_balance:.2f}")
        table.add_row("Running P/L", f"${self.running_pl:.2f}")
        table.add_row("Trade Count", str(self.trade_count))
        table.add_row("Win Rate", f"{self.calculate_win_rate():.1f}%")
        
        return table

    def create_trade_table(self):
        """Create rich table for recent trades"""
        table = Table(title="Recent Trades")
        table.add_column("Time", style="cyan")
        table.add_column("Signal", style="yellow")
        table.add_column("Price", style="blue")
        table.add_column("P/L", style="green")
        table.add_column("Running P/L", style="magenta")
        
        # Show last 5 trades
        for trade in self.simulator.trades[-5:]:
            pl = trade.get('pnl', 0)
            table.add_row(
                trade['timestamp'].strftime("%H:%M:%S"),
                trade['signal'],
                f"${trade['price']:.4f}",
                f"${pl:.2f}",
                f"${self.running_pl:.2f}"
            )
        
        return table

    def calculate_win_rate(self):
        """Calculate win rate from trades"""
        if not self.simulator.trades:
            return 0.0
        winning_trades = sum(1 for trade in self.simulator.trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(self.simulator.trades)) * 100

    async def run_simulation(self, token_pairs: list, duration_hours: int = 24):
        """Run simulation with live P/L tracking"""
        try:
            with Live(self.create_status_table(), refresh_per_second=1) as live:
                self.console.print("[bold green]Starting simulation...[/bold green]")
                
                async def update_display():
                    while True:
                        # Update tables
                        live.update(self.create_status_table())
                        self.console.print(self.create_trade_table())
                        await asyncio.sleep(1)

                # Start display update task
                display_task = asyncio.create_task(update_display())
                
                # Run simulation
                results = await self.simulator.run_live_simulation(
                    self.bot,
                    token_pairs,
                    duration_hours
                )
                
                # Stop display update
                display_task.cancel()
                
                # Print final results
                self.print_final_results(results)
                
                return results
                
        except Exception as e:
            self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
            raise

    def print_final_results(self, results: dict):
        """Print detailed final results"""
        table = Table(title="Simulation Final Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Initial Balance", f"${results['initial_balance']:.2f}")
        table.add_row("Final Balance", f"${results['final_balance']:.2f}")
        table.add_row("Total P/L", f"${results['final_balance'] - results['initial_balance']:.2f}")
        table.add_row("Return %", f"{((results['final_balance']/results['initial_balance'] - 1) * 100):.2f}%")
        table.add_row("Total Trades", str(len(results['trades'])))
        table.add_row("Win Rate", f"{self.calculate_win_rate():.1f}%")
        
        self.console.print(table)