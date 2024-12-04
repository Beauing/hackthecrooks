from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SuspiciousPatternDetector:
    def __init__(self):
        self.patterns = {
            'pump_and_dump': {
                'price_increase_threshold': 0.3,  # 30% sudden increase
                'volume_spike_threshold': 3,      # 3x normal volume
                'timeframe_hours': 24
            },
            'honeypot': {
                'buy_fail_rate': 0.2,            # 20% failed buys
                'sell_fail_rate': 0.4,           # 40% failed sells
                'min_transactions': 50           # Minimum sample size
            },
            'rug_pull': {
                'liquidity_drop_threshold': 0.5,  # 50% liquidity decrease
                'holder_drop_threshold': 0.3,     # 30% holder decrease
                'timeframe_hours': 12
            }
        }

    def detect_pump_and_dump(self, price_data: List[Dict], volume_data: List[Dict]) -> Dict:
        """Detect pump and dump patterns"""
        df = pd.DataFrame({
            'price': [p['price'] for p in price_data],
            'volume': [v['volume'] for v in volume_data],
            'timestamp': [p['timestamp'] for p in price_data]
        })
        
        # Calculate baseline metrics
        baseline_price = df['price'].rolling(window=24).mean()
        baseline_volume = df['volume'].rolling(window=24).mean()
        
        # Detect price pumps
        price_increases = df['price'].pct_change()
        volume_spikes = df['volume'] / baseline_volume
        
        suspicious_periods = []
        for i in range(len(df) - 24):
            window = df.iloc[i:i+24]
            if (max(price_increases.iloc[i:i+24]) > self.patterns['pump_and_dump']['price_increase_threshold'] and
                max(volume_spikes.iloc[i:i+24]) > self.patterns['pump_and_dump']['volume_spike_threshold']):
                suspicious_periods.append({
                    'start_time': window.iloc[0]['timestamp'],
                    'end_time': window.iloc[-1]['timestamp'],
                    'price_increase': max(price_increases.iloc[i:i+24]),
                    'volume_spike': max(volume_spikes.iloc[i:i+24])
                })
        
        return {
            'pattern_found': len(suspicious_periods) > 0,
            'suspicious_periods': suspicious_periods,
            'risk_score': min(1.0, len(suspicious_periods) * 0.2)
        }

    def detect_honeypot(self, transaction_data: List[Dict]) -> Dict:
        """Detect honeypot characteristics"""
        df = pd.DataFrame(transaction_data)
        
        if len(df) < self.patterns['honeypot']['min_transactions']:
            return {'pattern_found': False, 'risk_score': 0, 'reason': 'insufficient_data'}
        
        # Analyze transaction success rates
        buy_success_rate = len(df[df['type'] == 'buy' & df['success']]) / len(df[df['type'] == 'buy'])
        sell_success_rate = len(df[df['type'] == 'sell' & df['success']]) / len(df[df['type'] == 'sell'])
        
        # Analyze gas costs
        avg_buy_gas = df[df['type'] == 'buy']['gas_used'].mean()
        avg_sell_gas = df[df['type'] == 'sell']['gas_used'].mean()
        
        honeypot_indicators = {
            'high_sell_failure': sell_success_rate < (1 - self.patterns['honeypot']['sell_fail_rate']),
            'high_buy_failure': buy_success_rate < (1 - self.patterns['honeypot']['buy_fail_rate']),
            'asymmetric_gas': avg_sell_gas > (avg_buy_gas * 2)
        }
        
        risk_score = sum(1 for x in honeypot_indicators.values() if x) / len(honeypot_indicators)
        
        return {
            'pattern_found': risk_score > 0.5,
            'risk_score': risk_score,
            'indicators': honeypot_indicators,
            'metrics': {
                'buy_success_rate': buy_success_rate,
                'sell_success_rate': sell_success_rate,
                'avg_buy_gas': avg_buy_gas,
                'avg_sell_gas': avg_sell_gas
            }
        }

    def detect_rug_pull_risk(self, token_data: Dict) -> Dict:
        """Detect potential rug pull risk factors"""
        risk_factors = {
            'liquidity_risks': self.analyze_liquidity_risks(token_data),
            'ownership_risks': self.analyze_ownership_risks(token_data),
            'contract_risks': self.analyze_contract_risks(token_data)
        }
        
        risk_score = sum(r['risk_score'] for r in risk_factors.values()) / len(risk_factors)
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'high_risk': risk_score > 0.6,
            'recommendations': self.generate_risk_recommendations(risk_factors)
        }

    def analyze_liquidity_risks(self, token_data: Dict) -> Dict:
        """Analyze liquidity-related risks"""
        liquidity_history = token_data.get('liquidity_history', [])
        current_liquidity = token_data.get('current_liquidity', 0)
        
        risks = {
            'low_liquidity': current_liquidity < 50000,  # $50k minimum
            'declining_liquidity': False,
            'concentrated_lp': False
        }
        
        if liquidity_history:
            df = pd.DataFrame(liquidity_history)
            risks['declining_liquidity'] = (df['liquidity'].iloc[-1] / df['liquidity'].max()) < 0.7
            
        if 'lp_holders' in token_data:
            top_lp_holder_share = max(h['share'] for h in token_data['lp_holders'])
            risks['concentrated_lp'] = top_lp_holder_share > 0.3
            
        return {
            'risks': risks,
            'risk_score': sum(1 for r in risks.values() if r) / len(risks)
        }

    def analyze_ownership_risks(self, token_data: Dict) -> Dict:
        """Analyze ownership and control risks"""
        risks = {
            'high_owner_balance': False,
            'unlocked_tokens': False,
            'centralized_control': False
        }
        
        if 'owner_balance' in token_data:
            risks['high_owner_balance'] = token_data['owner_balance'] > 0.1  # >10% owned by creator
            
        if 'token_locks' in token_data:
            risks['unlocked_tokens'] = not any(lock['active'] for lock in token_data['token_locks'])
            
        if 'admin_keys' in token_data:
            risks['centralized_control'] = len(token_data['admin_keys']) < 2
            
        return {
            'risks': risks,
            'risk_score': sum(1 for r in risks.values() if r) / len(risks)
        }

    def analyze_contract_risks(self, token_data: Dict) -> Dict:
        """Analyze smart contract risks"""
        risks = {
            'unverified_contract': False,
            'dangerous_functions': False,
            'recent_deployment': False
        }
        
        contract_data = token_data.get('contract_data', {})
        
        risks['unverified_contract'] = not contract_data.get('is_verified', False)
        risks['dangerous_functions'] = any(f in contract_data.get('functions', [])
                                         for f in ['selfdestruct', 'delegatecall'])
        
        deployment_date = contract_data.get('deployment_date')
        if deployment_date:
            risks['recent_deployment'] = (datetime.now() - deployment_date).days < 7
            
        return {
            'risks': risks,
            'risk_score': sum(1 for r in risks.values() if r) / len(risks)
        }

    def generate_risk_recommendations(self, risk_factors: Dict) -> List[str]:
        """Generate recommendations based on risk analysis"""
        recommendations = []
        
        if risk_factors['liquidity_risks']['risk_score'] > 0.5:
            recommendations.append("Caution: Low or unstable liquidity detected")
            
        if risk_factors['ownership_risks']['risk_score'] > 0.5:
            recommendations.append("Warning: High ownership concentration or unlocked tokens")
            
        if risk_factors['contract_risks']['risk_score'] > 0.5:
            recommendations.append("Alert: Smart contract security concerns identified")
            
        return recommendations

    def get_overall_safety_score(self, token_data: Dict) -> float:
        """Calculate overall safety score for a token"""
        # Run all detection methods
        pump_dump = self.detect_pump_and_dump(
            token_data.get('price_history', []),
            token_data.get('volume_history', [])
        )
        honeypot = self.detect_honeypot(token_data.get('transactions', []))
        rug_pull = self.detect_rug_pull_risk(token_data)
        
        # Weight the different risk factors
        weights = {
            'pump_dump': 0.3,
            'honeypot': 0.4,
            'rug_pull': 0.3
        }
        
        safety_score = 1.0 - (
            pump_dump['risk_score'] * weights['pump_dump'] +
            honeypot['risk_score'] * weights['honeypot'] +
            rug_pull['risk_score'] * weights['rug_pull']
        )
        
        return max(0.0, min(1.0, safety_score))