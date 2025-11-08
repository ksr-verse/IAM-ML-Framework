"""
Visualization Module - Charts and Dashboards
Generates plots for decision probability, clusters, risk trends
Saves visualizations as PNG files
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Creates comprehensive visualizations for IAM data analysis.
    """
    
    def __init__(self, output_dir: str = 'outputs/visualizations'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all_visualizations(self, 
                                   features: pd.DataFrame,
                                   target: Optional[pd.Series] = None,
                                   original_data: Optional[pd.DataFrame] = None,
                                   insights: Optional[Dict] = None):
        """
        Generate all standard visualizations.
        
        Args:
            features: Processed features
            target: Target variable
            original_data: Original merged data
            insights: Generated insights dictionary
        """
        logger.info("=== Generating Visualizations ===")
        
        # Feature importance plots
        if insights and 'feature_importance' in insights:
            self._plot_feature_importance(insights['feature_importance'])
        
        # Target distribution (for classification)
        if target is not None:
            self._plot_target_distribution(target)
        
        # Risk analysis plots
        if original_data is not None and 'risk_score' in original_data.columns:
            self._plot_risk_trends(original_data)
        
        # Cluster visualization
        if insights and 'clusters' in insights:
            self._plot_cluster_distribution(insights['clusters'])
        
        # Access usage patterns
        if original_data is not None and 'frequency' in original_data.columns:
            self._plot_access_patterns(original_data)
        
        # Correlation heatmap
        self._plot_correlation_heatmap(features)
        
        # Feature distributions
        self._plot_feature_distributions(features)
        
        # Access removal recommendations
        if insights and 'access_reduction' in insights:
            self._plot_access_removal_recommendations(insights['access_reduction'], original_data)
        
        logger.info(f"=== Visualizations saved to {self.output_dir} ===")
    
    def _plot_feature_importance(self, importance_data: Dict):
        """
        Plot feature importance from models.
        
        Args:
            importance_data: Feature importance dictionary
        """
        logger.info("Creating feature importance plots...")
        
        for model_name, data in importance_data.items():
            if 'top_features' not in data:
                continue
            
            features_df = pd.DataFrame(data['top_features'])
            
            if len(features_df) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            plt.barh(features_df['feature'], features_df['importance'], color='steelblue')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title(f'Top 10 Feature Importance - {model_name.replace("_", " ").title()}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            filename = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved {filename}")
    
    def _plot_target_distribution(self, target: pd.Series):
        """
        Plot target variable distribution.
        
        Args:
            target: Target Series
        """
        logger.info("Creating target distribution plot...")
        
        plt.figure(figsize=(8, 6))
        target_counts = target.value_counts()
        
        # Bar plot
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color='coral')
        plt.xlabel('Target Class')
        plt.ylabel('Count')
        plt.title('Target Variable Distribution')
        plt.xticks(rotation=45)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('Set2'))
        plt.title('Target Variable Proportion')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'target_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_risk_trends(self, data: pd.DataFrame):
        """
        Plot risk score trends and distributions.
        
        Args:
            data: Original data with risk scores
        """
        logger.info("Creating risk trend plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall risk distribution
        axes[0, 0].hist(data['risk_score'], bins=30, color='salmon', edgecolor='black')
        axes[0, 0].axvline(data['risk_score'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {data["risk_score"].mean():.3f}')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Risk Score Distribution')
        axes[0, 0].legend()
        
        # 2. Risk by role (if available)
        if 'role' in data.columns:
            role_risk = data.groupby('role')['risk_score'].mean().sort_values(ascending=False).head(10)
            role_risk.plot(kind='barh', ax=axes[0, 1], color='orangered')
            axes[0, 1].set_xlabel('Average Risk Score')
            axes[0, 1].set_ylabel('Role')
            axes[0, 1].set_title('Risk Score by Role (Top 10)')
        else:
            axes[0, 1].text(0.5, 0.5, 'Role data not available', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
        
        # 3. Risk by department (if available)
        if 'department' in data.columns:
            dept_risk = data.groupby('department')['risk_score'].mean().sort_values(ascending=False)
            dept_risk.plot(kind='bar', ax=axes[1, 0], color='lightcoral')
            axes[1, 0].set_xlabel('Department')
            axes[1, 0].set_ylabel('Average Risk Score')
            axes[1, 0].set_title('Risk Score by Department')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Department data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # 4. Risk score box plot by decision (if available)
        if 'decision' in data.columns:
            data.boxplot(column='risk_score', by='decision', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Decision')
            axes[1, 1].set_ylabel('Risk Score')
            axes[1, 1].set_title('Risk Score Distribution by Decision')
            plt.sca(axes[1, 1])
            plt.xticks(rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Decision data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'risk_trends.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_cluster_distribution(self, cluster_data: Dict):
        """
        Plot cluster distribution and characteristics.
        
        Args:
            cluster_data: Cluster analysis dictionary
        """
        logger.info("Creating cluster distribution plots...")
        
        if 'distribution' not in cluster_data:
            logger.warning("  No cluster distribution data available")
            return
        
        distribution = cluster_data['distribution']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cluster size bar chart
        clusters = list(distribution.keys())
        sizes = list(distribution.values())
        
        axes[0].bar([f'Cluster {c}' for c in clusters], sizes, color='skyblue')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Cluster Size Distribution')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cluster pie chart
        axes[1].pie(sizes, labels=[f'Cluster {c}' for c in clusters], 
                   autopct='%1.1f%%', colors=sns.color_palette('Set3'))
        axes[1].set_title('Cluster Proportion')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'cluster_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_access_patterns(self, data: pd.DataFrame):
        """
        Plot access usage patterns.
        
        Args:
            data: Original data with access information
        """
        logger.info("Creating access pattern plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Access frequency distribution
        axes[0, 0].hist(data['frequency'], bins=30, color='mediumpurple', edgecolor='black')
        axes[0, 0].set_xlabel('Access Frequency')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Access Frequency Distribution')
        
        # 2. Top accessed items (if access_item exists)
        if 'access_item' in data.columns:
            top_access = data['access_item'].value_counts().head(15)
            top_access.plot(kind='barh', ax=axes[0, 1], color='mediumseagreen')
            axes[0, 1].set_xlabel('Number of Requests')
            axes[0, 1].set_ylabel('Access Item')
            axes[0, 1].set_title('Top 15 Most Requested Access Items')
        else:
            axes[0, 1].text(0.5, 0.5, 'Access item data not available', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
        
        # 3. Access frequency by role (if available)
        if 'role' in data.columns:
            role_freq = data.groupby('role')['frequency'].mean().sort_values(ascending=False).head(10)
            role_freq.plot(kind='bar', ax=axes[1, 0], color='darkorange')
            axes[1, 0].set_xlabel('Role')
            axes[1, 0].set_ylabel('Average Frequency')
            axes[1, 0].set_title('Average Access Frequency by Role')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Role data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # 4. Access type distribution (if available)
        if 'access_type' in data.columns:
            access_type_counts = data['access_type'].value_counts()
            access_type_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%',
                                   colors=sns.color_palette('Pastel1'))
            axes[1, 1].set_ylabel('')
            axes[1, 1].set_title('Access Type Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Access type data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'access_patterns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_correlation_heatmap(self, features: pd.DataFrame):
        """
        Plot correlation heatmap of features.
        
        Args:
            features: Feature DataFrame
        """
        logger.info("Creating correlation heatmap...")
        
        # Limit to top features if too many
        if len(features.columns) > 20:
            # Calculate variance and select top 20 features
            variances = features.var().sort_values(ascending=False)
            top_features = variances.head(20).index
            features_subset = features[top_features]
        else:
            features_subset = features
        
        plt.figure(figsize=(12, 10))
        correlation = features_subset.corr()
        
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0,
                   linewidths=0.5, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_feature_distributions(self, features: pd.DataFrame):
        """
        Plot distributions of key features.
        
        Args:
            features: Feature DataFrame
        """
        logger.info("Creating feature distribution plots...")
        
        # Select top 6 features by variance
        variances = features.var().sort_values(ascending=False)
        top_features = variances.head(6).index
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            axes[idx].hist(features[feature], bins=30, color='teal', 
                          alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{feature} Distribution')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'feature_distributions.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def _plot_access_removal_recommendations(self, access_reduction: Dict, 
                                            original_data: Optional[pd.DataFrame] = None):
        """
        Plot access removal recommendations - CRITICAL FOR BUSINESS!
        Shows which access should be removed and why.
        
        Args:
            access_reduction: Access reduction insights
            original_data: Original data for context
        """
        logger.info("Creating ACCESS REMOVAL RECOMMENDATIONS plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Low Usage Access Items (TOP CANDIDATES FOR REMOVAL)
        if access_reduction.get('low_usage_candidates'):
            low_usage = pd.DataFrame(access_reduction['low_usage_candidates']).head(15)
            
            axes[0, 0].barh(range(len(low_usage)), low_usage['avg_frequency'], 
                           color='red', alpha=0.7)
            axes[0, 0].set_yticks(range(len(low_usage)))
            axes[0, 0].set_yticklabels(low_usage['access_item'])
            axes[0, 0].set_xlabel('Average Usage Frequency')
            axes[0, 0].set_title('⚠️ LOW USAGE ACCESS - REMOVAL CANDIDATES\n(Top 15 - Used <5 times)', 
                                fontweight='bold', fontsize=12, color='red')
            axes[0, 0].axvline(5, color='black', linestyle='--', label='Threshold (5)')
            axes[0, 0].legend()
            axes[0, 0].invert_yaxis()
        else:
            axes[0, 0].text(0.5, 0.5, 'No low-usage access found', 
                           ha='center', va='center', fontsize=12)
            axes[0, 0].axis('off')
        
        # 2. Unused Access (90+ days) - IMMEDIATE REMOVAL
        if access_reduction.get('unused_access'):
            unused = pd.DataFrame(access_reduction['unused_access']).head(15)
            
            axes[0, 1].barh(range(len(unused)), unused['avg_days_since_use'], 
                           color='darkred', alpha=0.7)
            axes[0, 1].set_yticks(range(len(unused)))
            axes[0, 1].set_yticklabels(unused['access_item'])
            axes[0, 1].set_xlabel('Days Since Last Use')
            axes[0, 1].set_title('🚨 UNUSED ACCESS - IMMEDIATE REMOVAL\n(Top 15 - Not used 90+ days)', 
                                fontweight='bold', fontsize=12, color='darkred')
            axes[0, 1].axvline(90, color='black', linestyle='--', label='Threshold (90 days)')
            axes[0, 1].legend()
            axes[0, 1].invert_yaxis()
        else:
            axes[0, 1].text(0.5, 0.5, 'No unused access found', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
        
        # 3. Removal Impact (User count affected)
        if access_reduction.get('low_usage_candidates'):
            low_usage = pd.DataFrame(access_reduction['low_usage_candidates']).head(10)
            
            axes[1, 0].bar(range(len(low_usage)), low_usage['user_count'], 
                          color='orange', alpha=0.7)
            axes[1, 0].set_xticks(range(len(low_usage)))
            axes[1, 0].set_xticklabels(low_usage['access_item'], rotation=45, ha='right')
            axes[1, 0].set_ylabel('Number of Users Affected')
            axes[1, 0].set_title('📊 REMOVAL IMPACT ANALYSIS\n(How many users will be affected)', 
                                fontweight='bold', fontsize=12)
            axes[1, 0].grid(axis='y', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No impact data available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # 4. Summary Box - EXECUTIVE SUMMARY
        summary_text = f"""
        ACCESS REMOVAL RECOMMENDATIONS
        ═══════════════════════════════════════
        
        📌 TOTAL REMOVAL OPPORTUNITIES: {access_reduction.get('summary', 'N/A')}
        
        🔴 LOW USAGE ACCESS:
           {len(access_reduction.get('low_usage_candidates', []))} items used <5 times
           → Recommend: Review and remove unused entitlements
        
        🔴 UNUSED ACCESS (90+ days):
           {len(access_reduction.get('unused_access', []))} items not used recently
           → Recommend: Immediate removal/revocation
        
        💡 BUSINESS IMPACT:
           - Reduced security risk
           - Lower licensing costs
           - Simplified access management
           - Better compliance posture
        
        ⚠️ ACTION REQUIRED:
           Review highlighted access items with business owners
           for potential removal or policy update.
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, 
                       transform=axes[1, 1].transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'access_removal_recommendations.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")
    
    def plot_decision_probability(self, predictions_proba: np.ndarray, 
                                  labels: List[str]):
        """
        Plot decision probability distribution.
        
        Args:
            predictions_proba: Probability predictions from classifier
            labels: Class labels
        """
        logger.info("Creating decision probability plot...")
        
        plt.figure(figsize=(10, 6))
        
        for idx, label in enumerate(labels):
            plt.hist(predictions_proba[:, idx], bins=30, alpha=0.5, 
                    label=f'{label} Probability', edgecolor='black')
        
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Decision Probability Distribution')
        plt.legend()
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'decision_probability.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {filename}")


def main():
    """Test the visualizer."""
    from database import DatabaseConnector
    from preprocessing import DataPreprocessor
    from insights import InsightsGenerator
    
    # Load and preprocess data
    db = DatabaseConnector()
    db.connect()
    tables = db.fetch_all_tables()
    
    preprocessor = DataPreprocessor(db.get_ml_config(), db.get_merge_strategy())
    features, target = preprocessor.process_pipeline(tables)
    
    # Get original data
    cleaned_tables = {name: preprocessor.clean_dataframe(df, name) 
                     for name, df in tables.items()}
    original_data = preprocessor.merge_tables(cleaned_tables)
    
    # Generate insights
    insights_gen = InsightsGenerator(db.get_insights_config())
    insights = insights_gen.generate_all_insights(features, target, original_data)
    
    # Create visualizations
    viz = Visualizer()
    viz.generate_all_visualizations(features, target, original_data, insights)
    
    print(f"\nVisualizations saved to: {viz.output_dir}")
    
    db.disconnect()


if __name__ == "__main__":
    main()

