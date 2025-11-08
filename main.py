"""
Main Pipeline - IAM ML Framework Entry Point
Orchestrates the complete ML pipeline for IAM data analysis
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import DatabaseConnector
from preprocessing import DataPreprocessor
from model_training import ModelTrainer
from insights import InsightsGenerator
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iam_ml_framework.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IAMMLPipeline:
    """
    Main pipeline orchestrator for IAM ML analysis.
    """
    
    def __init__(self, db_config: str = 'config/db_config.yaml',
                 schema_config: str = 'config/schema_config.yaml'):
        """
        Initialize pipeline.
        
        Args:
            db_config: Path to database configuration
            schema_config: Path to schema configuration
        """
        logger.info("=" * 70)
        logger.info("IAM ML FRAMEWORK - Pipeline Initialization")
        logger.info("=" * 70)
        
        self.db = DatabaseConnector(db_config, schema_config)
        self.preprocessor = None
        self.trainer = None
        self.insights_gen = None
        self.visualizer = None
        
        # Data storage
        self.tables = {}
        self.features = None
        self.target = None
        self.original_data = None
        
    def load_data(self):
        """Load data from database or CSV files."""
        logger.info("\n[STEP 1/5] Loading Data...")
        
        if not self.db.connect():
            logger.error("Failed to connect to data source")
            return False
        
        self.tables = self.db.fetch_all_tables()
        
        if not self.tables:
            logger.error("No tables loaded")
            return False
        
        logger.info(f"✓ Successfully loaded {len(self.tables)} tables")
        for table_name, df in self.tables.items():
            logger.info(f"  - {table_name}: {len(df)} rows, {len(df.columns)} columns")
        
        return True
    
    def preprocess_data(self):
        """Preprocess and merge data."""
        logger.info("\n[STEP 2/5] Preprocessing Data...")
        
        ml_config = self.db.get_ml_config()
        merge_strategy = self.db.get_merge_strategy()
        
        self.preprocessor = DataPreprocessor(ml_config, merge_strategy)
        
        # Clean and merge tables
        cleaned_tables = {}
        for table_name, df in self.tables.items():
            cleaned_tables[table_name] = self.preprocessor.clean_dataframe(df, table_name)
        
        self.original_data = self.preprocessor.merge_tables(cleaned_tables)
        
        # Prepare features and target
        encoded_data = self.preprocessor.encode_features(self.original_data, fit=True)
        self.features, self.target = self.preprocessor.prepare_features(encoded_data)
        
        logger.info(f"✓ Preprocessing complete")
        logger.info(f"  - Features shape: {self.features.shape}")
        if self.target is not None:
            logger.info(f"  - Target shape: {self.target.shape}")
            logger.info(f"  - Target classes: {list(self.target.unique())}")
        else:
            logger.info(f"  - No target variable (unsupervised learning)")
        
        return True
    
    def train_models(self):
        """Train ML models."""
        logger.info("\n[STEP 3/5] Training Models...")
        
        ml_config = self.db.get_ml_config()
        self.trainer = ModelTrainer(ml_config)
        
        results = self.trainer.train(self.features, self.target)
        
        logger.info(f"✓ Training complete")
        logger.info(f"  - Models trained: {len(results['models'])}")
        
        # Print metrics summary
        for model_name, metrics in results['metrics'].items():
            logger.info(f"\n  {model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.4f}")
        
        return True
    
    def generate_insights(self):
        """Generate insights from trained models."""
        logger.info("\n[STEP 4/5] Generating Insights...")
        
        insights_config = self.db.get_insights_config()
        self.insights_gen = InsightsGenerator(insights_config)
        
        insights = self.insights_gen.generate_all_insights(
            self.features, 
            self.target, 
            self.original_data
        )
        
        # Generate and print report
        report = self.insights_gen.generate_report()
        print("\n" + report)
        
        logger.info(f"✓ Insights generation complete")
        
        return insights
    
    def create_visualizations(self, insights):
        """Create visualizations."""
        logger.info("\n[STEP 5/5] Creating Visualizations...")
        
        self.visualizer = Visualizer()
        self.visualizer.generate_all_visualizations(
            self.features,
            self.target,
            self.original_data,
            insights
        )
        
        logger.info(f"✓ Visualizations complete")
        logger.info(f"  - Saved to: {self.visualizer.output_dir}")
        
        return True
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: Preprocess
            if not self.preprocess_data():
                return False
            
            # Step 3: Train models
            if not self.train_models():
                return False
            
            # Step 4: Generate insights
            insights = self.generate_insights()
            
            # Step 5: Create visualizations
            self.create_visualizations(insights)
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Total execution time: {duration:.2f} seconds")
            logger.info(f"Models saved to: models/")
            logger.info(f"Insights saved to: outputs/insights/")
            logger.info(f"Visualizations saved to: outputs/visualizations/")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
        
        finally:
            self.db.disconnect()
    
    def run_training_only(self):
        """Run only data loading, preprocessing, and model training."""
        logger.info("Running TRAINING mode (no insights/visualizations)")
        
        try:
            if not self.load_data():
                return False
            
            if not self.preprocess_data():
                return False
            
            if not self.train_models():
                return False
            
            logger.info("\n✓ Training complete - Models saved to models/")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False
        
        finally:
            self.db.disconnect()
    
    def run_insights_only(self):
        """Generate insights from existing models and data."""
        logger.info("Running INSIGHTS mode (using existing data)")
        
        try:
            if not self.load_data():
                return False
            
            if not self.preprocess_data():
                return False
            
            insights = self.generate_insights()
            self.create_visualizations(insights)
            
            logger.info("\n✓ Insights and visualizations generated")
            return True
            
        except Exception as e:
            logger.error(f"Insights generation failed: {e}", exc_info=True)
            return False
        
        finally:
            self.db.disconnect()


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description='IAM ML Framework - Generic ML Pipeline for IAM Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (training + insights + visualizations)
  python main.py --mode full
  
  # Run training only (build models)
  python main.py --mode train
  
  # Generate insights from existing data (no training)
  python main.py --mode insights
  
  # Use custom config files
  python main.py --mode full --db-config config/my_db.yaml --schema-config config/my_schema.yaml
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'train', 'insights'],
        default='full',
        help='Pipeline mode: full (complete), train (training only), insights (insights only)'
    )
    
    parser.add_argument(
        '--db-config',
        type=str,
        default='config/db_config.yaml',
        help='Path to database configuration file'
    )
    
    parser.add_argument(
        '--schema-config',
        type=str,
        default='config/schema_config.yaml',
        help='Path to schema configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IAMMLPipeline(args.db_config, args.schema_config)
    
    # Run based on mode
    if args.mode == 'full':
        success = pipeline.run_full_pipeline()
    elif args.mode == 'train':
        success = pipeline.run_training_only()
    elif args.mode == 'insights':
        success = pipeline.run_insights_only()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

