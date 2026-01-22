"""
Machine Learning Predictor Module
Pendekatan Prediktif menggunakan algoritma ML
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """
    Kelas untuk prediksi menggunakan Machine Learning
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame dengan fitur teknikal
        """
        self.df = df.copy()
        self.models = {}
        self.best_model = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_features(self, target_col: str = 'Direction', 
                         exclude_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Menyiapkan fitur untuk training
        """
        df = self.df.copy()
        
        # Kolom yang tidak akan digunakan sebagai fitur
        if exclude_cols is None:
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                           'Log_Returns', 'Direction', 'Dividends', 'Stock Splits']
        
        # Shift target untuk prediksi hari berikutnya
        df['Target'] = df[target_col].shift(-1)
        df = df.dropna()
        
        # Pilih fitur
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col != 'Target']
        
        X = df[feature_cols]
        y = df['Target']
        
        # Hapus kolom dengan nilai NaN atau infinite
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how='any')
        
        # Sinkronkan index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        self.feature_cols = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Dict:
        """
        Melatih berbagai model ML
        """
        # Split data dengan mempertahankan urutan waktu
        split_idx = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Definisi model
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=0.1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=20,
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, use_label_encoder=False, 
                eval_metric='logloss'
            ),
            'SVM': SVC(
                kernel='rbf', probability=True, random_state=42, C=1.0
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            try:
                # Training
                if name in ['Logistic Regression', 'SVM']:
                    model.fit(X_train_scaled, self.y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                if name in ['Logistic Regression', 'SVM']:
                    cv_scores = cross_val_score(model, X_train_scaled, self.y_train, 
                                               cv=tscv, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                               cv=tscv, scoring='accuracy')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                self.models[name] = model
                
                print(f"✅ {name}: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        # Pilih model terbaik berdasarkan ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        # Feature importance untuk model tree-based
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results
    
    def create_ensemble(self) -> Dict:
        """
        Membuat model ensemble dari beberapa model terbaik
        """
        if not self.models:
            raise ValueError("Latih model terlebih dahulu dengan train_models()")
        
        # Buat voting classifier
        estimators = [
            ('rf', self.models.get('Random Forest')),
            ('gb', self.models.get('Gradient Boosting')),
            ('xgb', self.models.get('XGBoost'))
        ]
        estimators = [(name, model) for name, model in estimators if model is not None]
        
        if len(estimators) < 2:
            return {}
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['Ensemble'] = ensemble
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
    
    def predict_next_day(self, use_ensemble: bool = False) -> Dict:
        """
        Memprediksi probabilitas untuk hari berikutnya
        """
        # Ambil data terbaru
        latest_data = self.df.iloc[-1:][self.feature_cols]
        latest_data = latest_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        model = self.models.get('Ensemble') if use_ensemble else self.best_model
        model_name = 'Ensemble' if use_ensemble else self.best_model_name
        
        if model_name in ['Logistic Regression', 'SVM']:
            latest_scaled = self.scaler.transform(latest_data)
            prediction = model.predict(latest_scaled)[0]
            probability = model.predict_proba(latest_scaled)[0]
        else:
            prediction = model.predict(latest_data)[0]
            probability = model.predict_proba(latest_data)[0]
        
        return {
            'model_used': model_name,
            'prediction': 'NAIK 📈' if prediction == 1 else 'TURUN 📉',
            'probability_up': probability[1],
            'probability_down': probability[0],
            'confidence': max(probability),
            'latest_close': self.df['Close'].iloc[-1],
            'latest_date': str(self.df.index[-1].date()) if hasattr(self.df.index[-1], 'date') else str(self.df.index[-1])
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Mendapatkan perbandingan semua model
        """
        if not self.models:
            return pd.DataFrame()
        
        X, y = self.prepare_features()
        results = self.train_models(X, y)
        
        comparison = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [r['accuracy'] for r in results.values()],
            'Precision': [r['precision'] for r in results.values()],
            'Recall': [r['recall'] for r in results.values()],
            'F1-Score': [r['f1_score'] for r in results.values()],
            'ROC-AUC': [r['roc_auc'] for r in results.values()],
            'CV Mean': [r['cv_mean'] for r in results.values()],
            'CV Std': [r['cv_std'] for r in results.values()]
        })
        
        return comparison.sort_values('ROC-AUC', ascending=False)
    
    def get_feature_importance_report(self, top_n: int = 15) -> pd.DataFrame:
        """
        Mendapatkan fitur terpenting
        """
        if self.feature_importance is None:
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def backtest(self, initial_capital: float = 100000000) -> Dict:
        """
        Melakukan backtest sederhana pada data test
        """
        if self.X_test is None:
            raise ValueError("Latih model terlebih dahulu")
        
        model = self.best_model
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            X_test_scaled = self.scaler.transform(self.X_test)
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(self.X_test)
        
        # Simulasi trading
        test_returns = self.df.loc[self.y_test.index, 'Returns']
        
        # Strategy returns (beli jika prediksi naik)
        strategy_returns = predictions * test_returns
        
        # Cumulative returns
        cumulative_market = (1 + test_returns).cumprod()
        cumulative_strategy = (1 + strategy_returns).cumprod()
        
        # Metrics
        total_return_market = cumulative_market.iloc[-1] - 1
        total_return_strategy = cumulative_strategy.iloc[-1] - 1
        
        # Sharpe Ratio (assuming 252 trading days)
        sharpe_market = (test_returns.mean() / test_returns.std()) * np.sqrt(252)
        sharpe_strategy = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Win rate
        winning_trades = ((predictions == 1) & (test_returns > 0)).sum()
        total_trades = (predictions == 1).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Final values
        final_market = initial_capital * cumulative_market.iloc[-1]
        final_strategy = initial_capital * cumulative_strategy.iloc[-1]
        
        return {
            'initial_capital': initial_capital,
            'final_value_market': final_market,
            'final_value_strategy': final_strategy,
            'total_return_market': total_return_market,
            'total_return_strategy': total_return_strategy,
            'sharpe_market': sharpe_market,
            'sharpe_strategy': sharpe_strategy,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'outperformance': total_return_strategy - total_return_market,
            'cumulative_market': cumulative_market,
            'cumulative_strategy': cumulative_strategy
        }


# Tambahan: Typing support
from typing import Tuple, Dict