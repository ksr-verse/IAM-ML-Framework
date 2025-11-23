"""
Insights Module - Explainable AI Insights Generator
Provides feature importance, risk trends, access reduction opportunities
Outputs results as JSON and CSV
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightsGenerator:
    """
    Generates actionable insights from trained models and data.
    """
    
    def __init__(self, insights_config: dict, models_dir: str = 'models', 
                 output_dir: str = 'outputs/insights'):
        """
        Initialize insights generator.
        
        Args:
            insights_config: Configuration for insight generation
            models_dir: Directory containing trained models
            output_dir: Directory to save insights
        """
        self.insights_config = insights_config
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.insights = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _find_best_model(self) -> Optional[str]:
        """
        Automatically find the best trained model based on feature importance.
        
        Returns:
            Path to best model, or None if not found
        """
        # Look for all classifier models
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_classifier.pkl')]
        
        if not model_files:
            return None
        
        # Prefer models with feature importance (tree-based models)
        priority_order = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        for model_name in priority_order:
            for model_file in model_files:
                if model_name in model_file:
                    return os.path.join(self.models_dir, model_file)
        
        # Fallback: return first available
        return os.path.join(self.models_dir, model_files[0])
    
    def generate_all_insights(self, features: pd.DataFrame, 
                             target: Optional[pd.Series] = None,
                             original_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate all configured insights.
        
        Args:
            features: Processed feature DataFrame
            target: Target variable (if available)
            original_data: Original merged data before preprocessing
            
        Returns:
            Dictionary containing all insights
        """
        logger.info("=== Generating Insights ===")
        
        # Feature Importance
        if self.insights_config.get('generate_feature_importance', True):
            self.insights['feature_importance'] = self._generate_feature_importance()
        
        # Access Reduction Opportunities
        if self.insights_config.get('identify_access_reduction', True) and original_data is not None:
            self.insights['access_reduction'] = self._identify_access_reduction(original_data)
        
        # Risk Trends
        if original_data is not None:
            self.insights['risk_trends'] = self._analyze_risk_trends(original_data)
        
        # Anomaly Detection
        if self.insights_config.get('detect_anomalies', True):
            self.insights['anomalies'] = self._detect_anomalies(features, original_data)
        
        # Prediction Analysis (if classification model exists)
        if target is not None:
            self.insights['predictions'] = self._analyze_predictions(features, target)
        
        # Clustering Insights (if clustering model exists)
        self.insights['clusters'] = self._analyze_clusters(features, original_data)
        
        # Pattern Discovery (REAL DATA - for Part 2 post)
        if original_data is not None and target is not None:
            self.insights['pattern_discovery'] = self._discover_decision_patterns(original_data, target)
        
        # Save insights
        self._save_insights()
        
        # Pattern Discovery (NEW - for Part 2)
        if target is not None and original_data is not None:
            self.insights['pattern_discovery'] = self._discover_approval_patterns(original_data, target)
        
        logger.info("=== Insights Generation Complete ===")
        return self.insights
    
    def _generate_feature_importance(self) -> Dict[str, Any]:
        """
        Generate feature importance from trained models.
        
        Returns:
            Dictionary with feature importance analysis
        """
        logger.info("Generating feature importance insights...")
        
        importance_data = {}
        
        # Look for feature importance files
        for filename in os.listdir(self.models_dir):
            if 'feature_importance' in filename and filename.endswith('.csv'):
                model_name = filename.replace('_feature_importance.csv', '')
                importance_df = pd.read_csv(os.path.join(self.models_dir, filename))
                
                # Top 10 features
                top_features = importance_df.head(10).to_dict('records')
                importance_data[model_name] = {
                    'top_features': top_features,
                    'summary': f"Top predictor: {importance_df.iloc[0]['feature']} "
                              f"(importance: {importance_df.iloc[0]['importance']:.4f})"
                }
                
                logger.info(f"  {model_name}: {importance_data[model_name]['summary']}")
        
        return importance_data
    
    def _identify_access_reduction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify access items that can be reduced or removed.
        
        Args:
            data: Original merged DataFrame
            
        Returns:
            Dictionary with access reduction recommendations
        """
        logger.info("Identifying access reduction opportunities...")
        
        reduction_candidates = []
        low_usage_threshold = self.insights_config.get('low_usage_threshold', 5)
        
        # Check if required columns exist
        if 'frequency' in data.columns and 'access_item' in data.columns:
            # Find low-usage access items
            low_usage = data[data['frequency'] < low_usage_threshold]
            
            if len(low_usage) > 0:
                # Group by access item
                access_summary = low_usage.groupby('access_item').agg({
                    'user_id': 'count',
                    'frequency': 'mean'
                }).reset_index()
                
                access_summary.columns = ['access_item', 'user_count', 'avg_frequency']
                access_summary = access_summary.sort_values('user_count', ascending=False)
                
                reduction_candidates = access_summary.head(20).to_dict('records')
                
                logger.info(f"  Found {len(reduction_candidates)} access reduction candidates")
        
        # Check for unused access (if last_used_date exists)
        unused_access = []
        if 'last_used_date' in data.columns:
            data['last_used_date'] = pd.to_datetime(data['last_used_date'], errors='coerce')
            current_date = pd.Timestamp.now()
            data['days_since_use'] = (current_date - data['last_used_date']).dt.days
            
            # Access not used in 90+ days
            unused = data[data['days_since_use'] > 90]
            if len(unused) > 0 and 'access_item' in unused.columns:
                unused_summary = unused.groupby('access_item').agg({
                    'user_id': 'count',
                    'days_since_use': 'mean'
                }).reset_index()
                
                unused_summary.columns = ['access_item', 'user_count', 'avg_days_since_use']
                unused_summary = unused_summary.sort_values('user_count', ascending=False)
                
                unused_access = unused_summary.head(20).to_dict('records')
                
                logger.info(f"  Found {len(unused_access)} unused access items")
        
        return {
            'low_usage_candidates': reduction_candidates,
            'unused_access': unused_access,
            'summary': f"Total reduction opportunities: {len(reduction_candidates) + len(unused_access)}"
        }
    
    def _analyze_risk_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze risk score trends by role, department, etc.
        
        Args:
            data: Original merged DataFrame
            
        Returns:
            Dictionary with risk trend analysis
        """
        logger.info("Analyzing risk trends...")
        
        risk_analysis = {}
        risk_threshold = self.insights_config.get('risk_threshold', 0.7)
        
        if 'risk_score' not in data.columns:
            logger.warning("  No risk_score column found")
            return {'summary': 'Risk score data not available'}
        
        # Overall risk statistics
        risk_analysis['overall'] = {
            'mean_risk': float(data['risk_score'].mean()),
            'median_risk': float(data['risk_score'].median()),
            'high_risk_count': int((data['risk_score'] > risk_threshold).sum()),
            'high_risk_percentage': float((data['risk_score'] > risk_threshold).mean() * 100)
        }
        
        # Risk by role
        if 'role' in data.columns:
            role_risk = data.groupby('role')['risk_score'].agg(['mean', 'count']).reset_index()
            role_risk.columns = ['role', 'avg_risk', 'count']
            role_risk = role_risk.sort_values('avg_risk', ascending=False)
            risk_analysis['by_role'] = role_risk.head(10).to_dict('records')
        
        # Risk by department
        if 'department' in data.columns:
            dept_risk = data.groupby('department')['risk_score'].agg(['mean', 'count']).reset_index()
            dept_risk.columns = ['department', 'avg_risk', 'count']
            dept_risk = dept_risk.sort_values('avg_risk', ascending=False)
            risk_analysis['by_department'] = dept_risk.head(10).to_dict('records')
        
        # High-risk users
        if 'user_id' in data.columns:
            high_risk_users = data[data['risk_score'] > risk_threshold]
            if len(high_risk_users) > 0:
                user_risk = high_risk_users.groupby('user_id')['risk_score'].mean().reset_index()
                user_risk.columns = ['user_id', 'avg_risk']
                user_risk = user_risk.sort_values('avg_risk', ascending=False)
                risk_analysis['high_risk_users'] = user_risk.head(20).to_dict('records')
        
        logger.info(f"  High-risk items: {risk_analysis['overall']['high_risk_count']} "
                   f"({risk_analysis['overall']['high_risk_percentage']:.1f}%)")
        
        return risk_analysis
    
    def _detect_anomalies(self, features: pd.DataFrame, 
                         original_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            features: Processed features
            original_data: Original data for context
            
        Returns:
            Dictionary with anomaly analysis
        """
        logger.info("Detecting anomalies...")
        
        anomalies = {}
        
        # Statistical outliers (IQR method)
        outlier_indices = []
        for col in features.columns:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = features[(features[col] < lower_bound) | 
                                   (features[col] > upper_bound)].index
            outlier_indices.extend(col_outliers.tolist())
        
        # Get unique outlier indices
        unique_outliers = list(set(outlier_indices))
        
        anomalies['statistical_outliers'] = {
            'count': len(unique_outliers),
            'percentage': len(unique_outliers) / len(features) * 100
        }
        
        # Add context from original data if available
        if original_data is not None and len(unique_outliers) > 0:
            outlier_sample = original_data.iloc[unique_outliers[:10]]
            if 'user_id' in outlier_sample.columns:
                anomalies['sample_anomalies'] = outlier_sample['user_id'].tolist()
        
        logger.info(f"  Detected {len(unique_outliers)} anomalies "
                   f"({anomalies['statistical_outliers']['percentage']:.2f}%)")
        
        return anomalies
    
    def _analyze_predictions(self, features: pd.DataFrame, 
                            target: pd.Series) -> Dict[str, Any]:
        """
        Analyze prediction patterns from classification model.
        
        Args:
            features: Feature DataFrame
            target: Target variable
            
        Returns:
            Dictionary with prediction analysis
        """
        logger.info("Analyzing predictions...")
        
        prediction_analysis = {}
        
        # Automatically find and load BEST classification model
        model_path = self._find_best_model()
        if model_path and os.path.exists(model_path):
            logger.info(f"  Using best model: {os.path.basename(model_path)}")
            model = joblib.load(model_path)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Prediction distribution
            pred_dist = pd.Series(predictions).value_counts()
            prediction_analysis['distribution'] = pred_dist.to_dict()
            
            # Confidence analysis
            max_probs = probabilities.max(axis=1)
            prediction_analysis['confidence'] = {
                'mean_confidence': float(max_probs.mean()),
                'low_confidence_count': int((max_probs < 0.6).sum()),
                'low_confidence_percentage': float((max_probs < 0.6).mean() * 100)
            }
            
            logger.info(f"  Prediction distribution: {pred_dist.to_dict()}")
            logger.info(f"  Mean confidence: {prediction_analysis['confidence']['mean_confidence']:.2f}")
        else:
            logger.warning("  No classification model found")
            prediction_analysis = {'summary': 'Classification model not available'}
        
        return prediction_analysis
    
    def _analyze_clusters(self, features: pd.DataFrame,
                         original_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze clustering results and peer groups.
        
        Args:
            features: Feature DataFrame
            original_data: Original data for context
            
        Returns:
            Dictionary with cluster analysis
        """
        logger.info("Analyzing clusters...")
        
        cluster_analysis = {}
        
        # Load cluster assignments
        cluster_path = os.path.join(self.models_dir, 'cluster_assignments.csv')
        if os.path.exists(cluster_path):
            clusters = pd.read_csv(cluster_path)
            
            # Cluster distribution
            cluster_dist = clusters['cluster'].value_counts().sort_index()
            cluster_analysis['distribution'] = cluster_dist.to_dict()
            
            # Cluster characteristics
            features_with_clusters = features.copy()
            features_with_clusters['cluster'] = clusters['cluster'].values
            
            cluster_profiles = []
            for cluster_id in sorted(clusters['cluster'].unique()):
                cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
                
                profile = {
                    'cluster_id': int(cluster_id),
                    'size': len(cluster_data),
                    'feature_means': cluster_data.drop('cluster', axis=1).mean().to_dict()
                }
                cluster_profiles.append(profile)
            
            cluster_analysis['profiles'] = cluster_profiles
            
            logger.info(f"  Found {len(cluster_dist)} clusters")
            logger.info(f"  Cluster sizes: {cluster_dist.to_dict()}")
        else:
            logger.warning("  No clustering results found")
            cluster_analysis = {'summary': 'Clustering results not available'}
        
        return cluster_analysis
    
    def _discover_approval_patterns(self, data: pd.DataFrame, 
                                   target: pd.Series) -> Dict[str, Any]:
        """
        Discover actual approval/rejection patterns from data.
        Finds combinations like "High risk + Finance → 85% rejection"
        
        Args:
            data: Original merged DataFrame
            target: Target variable (decisions - encoded as 0=approve, 1=reject)
            
        Returns:
            Dictionary with discovered patterns
        """
        logger.info("Discovering approval patterns from data...")
        
        patterns = []
        
        # Add decision column for analysis (0=approve, 1=reject)
        analysis_data = data.copy()
        analysis_data['decision_encoded'] = target.values
        
        # Pattern 1: High risk + Department
        if 'risk_score' in analysis_data.columns and 'department' in analysis_data.columns:
            high_risk = analysis_data[analysis_data['risk_score'] > 0.7]
            if len(high_risk) > 0:
                dept_rejection = high_risk.groupby('department').apply(
                    lambda x: (x['decision_encoded'] == 1).sum() / len(x) * 100 if len(x) > 0 else 0
                ).sort_values(ascending=False)
                
                if len(dept_rejection) > 0 and dept_rejection.iloc[0] > 0:
                    top_dept = dept_rejection.index[0]
                    rejection_rate = dept_rejection.iloc[0]
                    patterns.append({
                        'pattern': f'High Risk (>0.7) + {top_dept} Dept',
                        'outcome': 'Rejection',
                        'probability': round(rejection_rate, 1),
                        'sample_size': len(high_risk[high_risk['department'] == top_dept])
                    })
        
        # Pattern 2: After-hours requests + Low frequency
        if 'timestamp_hour' in analysis_data.columns and 'frequency' in analysis_data.columns:
            after_hours = analysis_data[
                ((analysis_data['timestamp_hour'] < 6) | (analysis_data['timestamp_hour'] > 20)) &
                (analysis_data['frequency'] < 10)
            ]
            if len(after_hours) > 10:
                rejection_rate = (after_hours['decision_encoded'] == 1).sum() / len(after_hours) * 100
                patterns.append({
                    'pattern': 'After Hours (Before 6AM/After 8PM) + Low Frequency',
                    'outcome': 'Rejection',
                    'probability': round(rejection_rate, 1),
                    'sample_size': len(after_hours)
                })
        
        # Pattern 3: Manager role + Low risk
        if 'requester_role' in analysis_data.columns and 'risk_score' in analysis_data.columns:
            manager_low_risk = analysis_data[
                (analysis_data['requester_role'] == 'Manager') &
                (analysis_data['risk_score'] < 0.3)
            ]
            if len(manager_low_risk) > 10:
                approval_rate = (manager_low_risk['decision_encoded'] == 0).sum() / len(manager_low_risk) * 100
                patterns.append({
                    'pattern': 'Manager Role + Low Risk (<0.3)',
                    'outcome': 'Approval',
                    'probability': round(approval_rate, 1),
                    'sample_size': len(manager_low_risk)
                })
        
        # Pattern 4: Business hours + IT department
        if 'timestamp_hour' in analysis_data.columns and 'department' in analysis_data.columns:
            business_hours_it = analysis_data[
                (analysis_data['timestamp_hour'] >= 9) &
                (analysis_data['timestamp_hour'] <= 17) &
                (analysis_data['department'] == 'IT')
            ]
            if len(business_hours_it) > 10:
                approval_rate = (business_hours_it['decision_encoded'] == 0).sum() / len(business_hours_it) * 100
                patterns.append({
                    'pattern': 'Business Hours (9-17) + IT Dept',
                    'outcome': 'Approval',
                    'probability': round(approval_rate, 1),
                    'sample_size': len(business_hours_it)
                })
        
        logger.info(f"  Discovered {len(patterns)} patterns from data")
        
        return {
            'patterns': patterns,
            'summary': f'Found {len(patterns)} significant approval patterns'
        }
    
    def _save_insights(self):
        """Save all insights to JSON and CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f'insights_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)
        logger.info(f"Saved insights to {json_path}")
        
        # Save key insights as CSV
        csv_data = []
        
        # Access reduction opportunities
        if 'access_reduction' in self.insights:
            for candidate in self.insights['access_reduction'].get('low_usage_candidates', []):
                csv_data.append({
                    'insight_type': 'access_reduction',
                    'category': 'low_usage',
                    'item': candidate.get('access_item', 'N/A'),
                    'value': candidate.get('avg_frequency', 0),
                    'impact': candidate.get('user_count', 0)
                })
        
        # High-risk items
        if 'risk_trends' in self.insights and 'high_risk_users' in self.insights['risk_trends']:
            for user in self.insights['risk_trends']['high_risk_users'][:10]:
                csv_data.append({
                    'insight_type': 'risk',
                    'category': 'high_risk_user',
                    'item': user.get('user_id', 'N/A'),
                    'value': user.get('avg_risk', 0),
                    'impact': 'high'
                })
        
        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.output_dir, f'insights_summary_{timestamp}.csv')
            csv_df.to_csv(csv_path, index=False)
            logger.info(f"Saved insights summary to {csv_path}")
    
    def _discover_decision_patterns(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Discover REAL decision patterns from actual data.
        Finds combinations that lead to high rejection/approval rates.
        
        Args:
            data: Original merged DataFrame
            target: Target decisions
            
        Returns:
            Dictionary with discovered patterns
        """
        logger.info("Discovering decision patterns from real data...")
        
        patterns = []
        
        # Add target to data for analysis
        data_with_target = data.copy()
        data_with_target['decision_target'] = target
        
        # Pattern 1: High Risk + Department → Rejection Rate
        if 'risk_score' in data.columns and 'department' in data.columns:
            high_risk = data_with_target[data_with_target['risk_score'] > 0.7]
            if len(high_risk) > 0:
                for dept in high_risk['department'].unique():
                    dept_data = high_risk[high_risk['department'] == dept]
                    if len(dept_data) >= 10:  # At least 10 samples
                        reject_rate = (dept_data['decision_target'] == 1).mean() * 100
                        if reject_rate > 60:  # Significant rejection pattern
                            patterns.append({
                                'pattern': f"High Risk (>0.7) + {dept} Dept",
                                'outcome': 'Rejection',
                                'probability': reject_rate,
                                'sample_size': len(dept_data)
                            })
        
        # Pattern 2: After-hours requests (late night/early morning)
        if 'timestamp' in data.columns:
            data_with_target['hour'] = pd.to_datetime(data_with_target['timestamp']).dt.hour
            after_hours = data_with_target[(data_with_target['hour'] < 6) | (data_with_target['hour'] > 22)]
            if len(after_hours) >= 10:
                reject_rate = (after_hours['decision_target'] == 1).mean() * 100
                patterns.append({
                    'pattern': 'After-hours requests (10PM-6AM)',
                    'outcome': 'Rejection',
                    'probability': reject_rate,
                    'sample_size': len(after_hours)
                })
        
        # Pattern 3: Role + Risk combination
        if 'requester_role' in data.columns and 'risk_score' in data.columns:
            for role in data_with_target['requester_role'].unique():
                low_risk_role = data_with_target[
                    (data_with_target['requester_role'] == role) & 
                    (data_with_target['risk_score'] < 0.3)
                ]
                if len(low_risk_role) >= 10:
                    approval_rate = (low_risk_role['decision_target'] == 0).mean() * 100
                    if approval_rate > 80:  # High approval pattern
                        patterns.append({
                            'pattern': f"{role} Role + Low Risk (<0.3)",
                            'outcome': 'Approval',
                            'probability': approval_rate,
                            'sample_size': len(low_risk_role)
                        })
        
        # Pattern 4: Frequency + Access Type
        if 'frequency' in data.columns and 'access_type' in data.columns:
            low_freq = data_with_target[data_with_target['frequency'] < 5]
            if len(low_freq) >= 10:
                reject_rate = (low_freq['decision_target'] == 1).mean() * 100
                if reject_rate > 60:
                    patterns.append({
                        'pattern': 'Low Usage Frequency (<5)',
                        'outcome': 'Rejection',
                        'probability': reject_rate,
                        'sample_size': len(low_freq)
                    })
        
        # Sort by probability (strongest patterns first)
        patterns = sorted(patterns, key=lambda x: abs(x['probability'] - 50), reverse=True)
        
        logger.info(f"  Discovered {len(patterns)} real decision patterns")
        
        return {
            'patterns': patterns[:10],  # Top 10
            'summary': f"{len(patterns)} decision patterns discovered from real data"
        }
    
    def generate_report(self) -> str:
        """
        Generate a human-readable text report.
        
        Returns:
            Report string
        """
        report = ["=" * 60]
        report.append("IAM INSIGHTS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Feature Importance
        if 'feature_importance' in self.insights:
            report.append("FEATURE IMPORTANCE")
            report.append("-" * 60)
            for model, data in self.insights['feature_importance'].items():
                report.append(f"\n{model}:")
                report.append(f"  {data.get('summary', 'N/A')}")
        
        # Access Reduction
        if 'access_reduction' in self.insights:
            report.append("\nACCESS REDUCTION OPPORTUNITIES")
            report.append("-" * 60)
            report.append(self.insights['access_reduction'].get('summary', 'N/A'))
        
        # Risk Trends
        if 'risk_trends' in self.insights and 'overall' in self.insights['risk_trends']:
            report.append("\nRISK ANALYSIS")
            report.append("-" * 60)
            overall = self.insights['risk_trends']['overall']
            report.append(f"  Mean Risk Score: {overall['mean_risk']:.3f}")
            report.append(f"  High-Risk Items: {overall['high_risk_count']} "
                         f"({overall['high_risk_percentage']:.1f}%)")
        
        # Anomalies
        if 'anomalies' in self.insights:
            report.append("\nANOMALY DETECTION")
            report.append("-" * 60)
            if 'statistical_outliers' in self.insights['anomalies']:
                outliers = self.insights['anomalies']['statistical_outliers']
                report.append(f"  Detected: {outliers['count']} anomalies "
                             f"({outliers['percentage']:.2f}%)")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'insights_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved report to {report_path}")
        
        return report_text


def main():
    """Test the insights generator."""
    from database import DatabaseConnector
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    db = DatabaseConnector()
    db.connect()
    tables = db.fetch_all_tables()
    
    preprocessor = DataPreprocessor(db.get_ml_config(), db.get_merge_strategy())
    features, target = preprocessor.process_pipeline(tables)
    
    # Get original merged data for context
    cleaned_tables = {name: preprocessor.clean_dataframe(df, name) 
                     for name, df in tables.items()}
    original_data = preprocessor.merge_tables(cleaned_tables)
    
    # Generate insights
    insights_gen = InsightsGenerator(db.get_insights_config())
    insights = insights_gen.generate_all_insights(features, target, original_data)
    
    # Print report
    report = insights_gen.generate_report()
    print(report)
    
    db.disconnect()


if __name__ == "__main__":
    main()

