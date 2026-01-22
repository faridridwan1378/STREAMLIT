"""
Statistical Analysis Module
Copyright © 2026 Farid Ridwan | farid.rdwan@gmail.com
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple


class StatisticalAnalyzer:
    """Kelas untuk analisis statistik probabilitas saham"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
    
    def calculate_basic_probability(self) -> Dict:
        """Menghitung probabilitas dasar naik/turun"""
        if 'Returns' not in self.df.columns:
            return {}
            
        returns = self.df['Returns'].dropna()
        total_days = len(returns)
        
        if total_days == 0:
            return {}
        
        up_days = (returns > 0).sum()
        down_days = (returns < 0).sum()
        
        prob_up = up_days / total_days
        prob_down = down_days / total_days
        
        result = {
            'total_days': total_days,
            'up_days': int(up_days),
            'down_days': int(down_days),
            'prob_up': prob_up,
            'prob_down': prob_down,
            'avg_return': returns.mean(),
            'std_return': returns.std(),
        }
        
        self.results['basic'] = result
        return result
    
    def calculate_conditional_probability(self) -> Dict:
        """Probabilitas bersyarat"""
        if 'Direction' not in self.df.columns:
            return {}
            
        df = self.df.copy()
        df['Tomorrow'] = df['Direction'].shift(-1)
        df = df.dropna()
        
        up_today = df[df['Direction'] == 1]
        down_today = df[df['Direction'] == 0]
        
        prob_up_after_up = up_today['Tomorrow'].mean() if len(up_today) > 0 else 0
        prob_up_after_down = down_today['Tomorrow'].mean() if len(down_today) > 0 else 0
        
        streak = self._calculate_streak(df)
        
        result = {
            'prob_up_after_up': prob_up_after_up,
            'prob_down_after_up': 1 - prob_up_after_up,
            'prob_up_after_down': prob_up_after_down,
            'prob_down_after_down': 1 - prob_up_after_down,
            'streak_analysis': streak
        }
        
        self.results['conditional'] = result
        return result
    
    def _calculate_streak(self, df: pd.DataFrame) -> Dict:
        """Analisis streak"""
        streak_data = {}
        try:
            for streak_len in [2, 3]:
                up_mask = df['Direction'].rolling(streak_len).sum() == streak_len
                if up_mask.sum() > 0:
                    streak_data[f'after_{streak_len}_up'] = {
                        'prob_up': df.loc[up_mask, 'Tomorrow'].mean(),
                        'count': int(up_mask.sum())
                    }
        except:
            pass
        return streak_data
    
    def calculate_day_of_week_probability(self) -> pd.DataFrame:
        """Probabilitas per hari dalam seminggu"""
        if 'DayOfWeek' not in self.df.columns:
            return pd.DataFrame()
        
        day_names = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat']
        
        day_stats = self.df.groupby('DayOfWeek').agg({
            'Direction': ['mean', 'count'],
            'Returns': ['mean', 'std']
        })
        
        day_stats.columns = ['Prob_Naik', 'Jumlah_Hari', 'Avg_Return', 'Std_Return']
        day_stats.index = [day_names[i] if i < len(day_names) else f'Day_{i}' for i in day_stats.index]
        
        self.results['day_of_week'] = day_stats
        return day_stats
    
    def run_full_analysis(self) -> Dict:
        """Jalankan semua analisis"""
        self.calculate_basic_probability()
        self.calculate_conditional_probability()
        self.calculate_day_of_week_probability()
        return self.results