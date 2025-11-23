"""
Model Training Module - Generic ML Pipeline
Automatically detects target columns
Runs classification, clustering, or regression
Saves trained models and prints metrics
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import joblib
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score
)

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced ML algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available - install with: pip install lightgbm")


class ModelTrainer:
    """
    Generic ML trainer that handles multiple model types:
    - Classification (with target variable)
    - Clustering (without target variable)
    - Regression (continuous target)
    """
    
    def __init__(self, ml_config: dict, models_dir: str = 'models'):
        """
        Initialize model trainer.
        
        Args:
            ml_config: ML configuration from schema
            models_dir: Directory to save trained models
        """
        self.ml_config = ml_config
        self.models_dir = models_dir
        self.trained_models = {}
        self.metrics = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def train(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Main training pipeline - automatically selects appropriate model type.
        
        Args:
            features: Feature DataFrame
            target: Target Series (None for unsupervised learning)
            
        Returns:
            Dictionary with trained models and metrics
        """
        logger.info("=== Starting Model Training ===")
        
        if target is not None and not target.empty:
            # Supervised learning
            if self._is_classification_task(target):
                logger.info("Detected classification task")
                self._train_classification(features, target)
            else:
                logger.info("Detected regression task")
                self._train_regression(features, target)
        else:
            # Unsupervised learning
            logger.info("No target variable - running clustering")
            self._train_clustering(features)
        
        logger.info("=== Training Complete ===")
        return {
            'models': self.trained_models,
            'metrics': self.metrics
        }
    
    def _is_classification_task(self, target: pd.Series) -> bool:
        """
        Determine if task is classification or regression.
        
        Args:
            target: Target variable
            
        Returns:
            True if classification, False if regression
        """
        # If categorical or few unique values, it's classification
        if target.dtype == 'object' or target.dtype.name == 'category':
            return True
        
        unique_ratio = len(target.unique()) / len(target)
        return unique_ratio < 0.05  # Less than 5% unique values
    
    def _train_classification(self, features: pd.DataFrame, target: pd.Series):
        """
        Train classification models.
        
        Args:
            features: Feature DataFrame
            target: Target Series
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Get algorithms from config (respect priority order)
        config_algorithms = []
        if self.ml_config and 'models' in self.ml_config:
            classification_config = self.ml_config['models'].get('classification', {})
            if classification_config.get('enabled', True):
                config_algorithms = classification_config.get('algorithms', [])
        
        # If no config, use default priority order
        if not config_algorithms:
            config_algorithms = ['random_forest', 'gradient_boosting', 'lightgbm', 'xgboost', 'logistic_regression']
        
        logger.info(f"Training algorithms (priority order): {config_algorithms}")
        
        # Define all available models
        available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            available_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        else:
            if 'xgboost' in config_algorithms:
                logger.warning("XGBoost requested in config but not available. Install with: pip install xgboost")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            available_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        else:
            if 'lightgbm' in config_algorithms:
                logger.warning("LightGBM requested in config but not available. Install with: pip install lightgbm")
        
        # Filter models based on config (respect priority order)
        models_to_train = {}
        for algo in config_algorithms:
            if algo in available_models:
                models_to_train[algo] = available_models[algo]
            else:
                logger.warning(f"Algorithm '{algo}' requested in config but not available/implemented")
        
        if not models_to_train:
            logger.error("No models available to train!")
            return
        
        # Train each model in priority order
        for model_name, model in models_to_train.items():
            logger.info(f"\nTraining {model_name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Handle binary vs multiclass
                average = 'binary' if len(target.unique()) == 2 else 'weighted'
                precision = precision_score(y_test, y_pred, average=average, zero_division=0)
                recall = recall_score(y_test, y_pred, average=average, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Store model and metrics
                self.trained_models[model_name] = model
                self.metrics[model_name] = metrics
                
                # Log results
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
                # Save model
                model_path = os.path.join(self.models_dir, f'{model_name}_classifier.pkl')
                joblib.dump(model, model_path)
                logger.info(f"  Saved model to {model_path}")
                
                # Save feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': features.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_path = os.path.join(self.models_dir, f'{model_name}_feature_importance.csv')
                    importance.to_csv(importance_path, index=False)
                    logger.info(f"  Saved feature importance to {importance_path}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
    
    def _train_regression(self, features: pd.DataFrame, target: pd.Series):
        """
        Train regression models.
        
        Args:
            features: Feature DataFrame
            target: Target Series
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.ensemble import RandomForestRegressor
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train each model
        for model_name, model in models.items():
            logger.info(f"\nTraining {model_name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                }
                
                # Store model and metrics
                self.trained_models[model_name] = model
                self.metrics[model_name] = metrics
                
                # Log results
                logger.info(f"  RMSE: {rmse:.4f}")
                logger.info(f"  MAE: {mae:.4f}")
                logger.info(f"  R²: {r2:.4f}")
                
                # Save model
                model_path = os.path.join(self.models_dir, f'{model_name}_regressor.pkl')
                joblib.dump(model, model_path)
                logger.info(f"  Saved model to {model_path}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
    
    def _train_clustering(self, features: pd.DataFrame):
        """
        Train clustering models.
        
        Args:
            features: Feature DataFrame
        """
        logger.info("Training clustering models...")
        
        # Determine optimal number of clusters for KMeans
        optimal_k = self._find_optimal_clusters(features)
        
        # KMeans
        try:
            logger.info(f"\nTraining KMeans with k={optimal_k}...")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            silhouette = silhouette_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            
            metrics = {
                'n_clusters': optimal_k,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
            }
            
            # Store model and metrics
            self.trained_models['kmeans'] = kmeans
            self.metrics['kmeans'] = metrics
            
            # Log results
            logger.info(f"  Silhouette Score: {silhouette:.4f}")
            logger.info(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
            logger.info(f"  Cluster sizes: {metrics['cluster_sizes']}")
            
            # Save model
            model_path = os.path.join(self.models_dir, 'kmeans_clustering.pkl')
            joblib.dump(kmeans, model_path)
            logger.info(f"  Saved model to {model_path}")
            
            # Save cluster assignments
            cluster_df = pd.DataFrame({
                'cluster': cluster_labels
            }, index=features.index)
            cluster_path = os.path.join(self.models_dir, 'cluster_assignments.csv')
            cluster_df.to_csv(cluster_path)
            logger.info(f"  Saved cluster assignments to {cluster_path}")
            
        except Exception as e:
            logger.error(f"Error training KMeans: {e}")
    
    def _find_optimal_clusters(self, features: pd.DataFrame, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            features: Feature DataFrame
            max_k: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        max_k = min(max_k, len(features) // 2)  # Can't have more clusters than half the samples
        
        if max_k < 2:
            return 2
        
        inertias = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Find elbow (largest decrease)
        diffs = np.diff(inertias)
        optimal_k = int(np.argmin(diffs) + 2)  # +2 because we start from k=2
        
        logger.info(f"  Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model file (without .pkl extension)
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on metrics.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.metrics:
            logger.warning("No trained models available")
            return None, None
        
        # For classification, use accuracy
        if 'accuracy' in list(self.metrics.values())[0]:
            best_model_name = max(self.metrics.items(), 
                                 key=lambda x: x[1].get('accuracy', 0))[0]
        # For regression, use R²
        elif 'r2_score' in list(self.metrics.values())[0]:
            best_model_name = max(self.metrics.items(), 
                                 key=lambda x: x[1].get('r2_score', -999))[0]
        # For clustering, use silhouette score
        else:
            best_model_name = max(self.metrics.items(), 
                                 key=lambda x: x[1].get('silhouette_score', -1))[0]
        
        logger.info(f"Best model: {best_model_name}")
        return best_model_name, self.trained_models[best_model_name]


def main():
    """Test the model trainer."""
    from database import DatabaseConnector
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    db = DatabaseConnector()
    db.connect()
    tables = db.fetch_all_tables()
    
    preprocessor = DataPreprocessor(db.get_ml_config(), db.get_merge_strategy())
    features, target = preprocessor.process_pipeline(tables)
    
    # Train models
    trainer = ModelTrainer(db.get_ml_config())
    results = trainer.train(features, target)
    
    # Print summary
    print("\n=== Training Summary ===")
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {metric}: {value}")
    
    db.disconnect()


if __name__ == "__main__":
    main()

