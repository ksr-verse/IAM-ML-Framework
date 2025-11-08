"""
Preprocessing Module - Data Cleaning and Merging
Takes DataFrames from database module
Performs standard cleaning, date parsing, encoding
Joins data based on common keys
Returns training-ready dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Flexible data preprocessor that automatically handles:
    - Missing value imputation
    - Date parsing and feature engineering
    - Categorical encoding
    - Data merging based on shared keys
    - Feature scaling
    """
    
    def __init__(self, ml_config: dict, merge_strategy: Optional[dict] = None):
        """
        Initialize preprocessor with ML configuration.
        
        Args:
            ml_config: ML configuration from schema_config.yaml
            merge_strategy: Merge strategy configuration
        """
        self.ml_config = ml_config
        self.merge_strategy = merge_strategy
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_dataframe(self, df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
        """
        Clean a single DataFrame.
        
        Args:
            df: Input DataFrame
            table_name: Name of the table (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {table_name} - Shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Parse date columns
        df_clean = self._parse_dates(df_clean)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        logger.info(f"Cleaned {table_name} - Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values intelligently based on column type.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df_filled = df.copy()
        
        for col in df_filled.columns:
            missing_count = df_filled[col].isnull().sum()
            if missing_count > 0:
                # Numerical columns: fill with median
                if df_filled[col].dtype in ['int64', 'float64']:
                    fill_value = df_filled[col].median()
                    df_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"  Filled {missing_count} missing values in {col} with median: {fill_value}")
                
                # Categorical columns: fill with mode or 'Unknown'
                else:
                    if df_filled[col].mode().empty:
                        fill_value = 'Unknown'
                    else:
                        fill_value = df_filled[col].mode()[0]
                    df_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"  Filled {missing_count} missing values in {col} with: {fill_value}")
        
        return df_filled
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns and create derived features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with parsed dates and derived features
        """
        date_columns = self.ml_config.get('date_columns', [])
        df_parsed = df.copy()
        
        for col in date_columns:
            if col in df_parsed.columns:
                try:
                    # Parse to datetime
                    df_parsed[col] = pd.to_datetime(df_parsed[col], errors='coerce')
                    
                    # Create derived features
                    df_parsed[f'{col}_year'] = df_parsed[col].dt.year
                    df_parsed[f'{col}_month'] = df_parsed[col].dt.month
                    df_parsed[f'{col}_day_of_week'] = df_parsed[col].dt.dayofweek
                    df_parsed[f'{col}_hour'] = df_parsed[col].dt.hour
                    
                    # Calculate days since epoch (for relative time)
                    df_parsed[f'{col}_days_since'] = (df_parsed[col] - pd.Timestamp('2020-01-01')).dt.days
                    
                    logger.info(f"  Parsed date column: {col} and created derived features")
                    
                except Exception as e:
                    logger.warning(f"  Could not parse date column {col}: {e}")
        
        return df_parsed
    
    def merge_tables(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple tables based on merge strategy.
        
        Args:
            tables: Dictionary of table_name -> DataFrame
            
        Returns:
            Merged DataFrame
        """
        if not self.merge_strategy:
            logger.warning("No merge strategy defined, using first table only")
            return list(tables.values())[0]
        
        base_table_name = self.merge_strategy['base_table']
        if base_table_name not in tables:
            logger.error(f"Base table {base_table_name} not found")
            return pd.DataFrame()
        
        merged_df = tables[base_table_name].copy()
        logger.info(f"Starting merge with base table: {base_table_name} ({len(merged_df)} rows)")
        
        # Perform joins
        joins = self.merge_strategy.get('joins', [])
        for join_config in joins:
            # Ensure join_config is a dictionary
            if not isinstance(join_config, dict):
                logger.warning(f"Invalid join configuration: {join_config}, skipping")
                continue
                
            table_name = join_config.get('table')
            if not table_name:
                logger.warning(f"Join configuration missing 'table' key, skipping")
                continue
                
            if table_name not in tables:
                logger.warning(f"Table {table_name} not found, skipping join")
                continue
            
            on_columns = join_config.get('on')
            if not on_columns:
                logger.warning(f"Join configuration for {table_name} missing 'on' key, skipping")
                continue
                
            how = join_config.get('how', 'left')
            
            # Check if join columns exist
            if not all(col in merged_df.columns for col in on_columns):
                logger.warning(f"Join columns {on_columns} not found in merged data, skipping")
                continue
            if not all(col in tables[table_name].columns for col in on_columns):
                logger.warning(f"Join columns {on_columns} not found in {table_name}, skipping")
                continue
            
            # Perform merge
            before_rows = len(merged_df)
            merged_df = merged_df.merge(
                tables[table_name],
                on=on_columns,
                how=how,
                suffixes=('', f'_{table_name}')
            )
            logger.info(f"  Joined {table_name} on {on_columns} ({how}): {before_rows} -> {len(merged_df)} rows")
        
        logger.info(f"Final merged dataset: {merged_df.shape}")
        return merged_df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit new encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        categorical_columns = self.ml_config.get('categorical_columns', [])
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                try:
                    if fit:
                        # Fit new encoder
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        self.label_encoders[col] = le
                        logger.info(f"  Encoded {col}: {len(le.classes_)} unique values")
                    else:
                        # Use existing encoder
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Handle unknown categories
                            df_encoded[col] = df_encoded[col].astype(str).map(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
                        else:
                            logger.warning(f"No encoder found for {col}, skipping")
                            
                except Exception as e:
                    logger.warning(f"Could not encode {col}: {e}")
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target for ML.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        target_column = self.ml_config.get('target_column')
        exclude_columns = self.ml_config.get('exclude_columns', [])
        
        # Extract target if exists
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column].copy()
            logger.info(f"Target column: {target_column}")
            logger.info(f"  Target distribution:\n{target.value_counts()}")
        
        # Prepare features
        feature_columns = [col for col in df.columns 
                          if col != target_column and col not in exclude_columns]
        
        features = df[feature_columns].copy()
        
        # Remove any remaining non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        removed_cols = set(features.columns) - set(numeric_features.columns)
        if removed_cols:
            logger.info(f"  Removed non-numeric columns: {removed_cols}")
        
        # Handle any remaining NaN or inf values
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        logger.info(f"Final features shape: {numeric_features.shape}")
        logger.info(f"Feature columns: {list(numeric_features.columns)}")
        
        return numeric_features, target
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler.
        
        Args:
            features: Feature DataFrame
            fit: Whether to fit new scaler
            
        Returns:
            Scaled features DataFrame
        """
        if fit:
            scaled_values = self.scaler.fit_transform(features)
            logger.info("Fitted and transformed features with StandardScaler")
        else:
            scaled_values = self.scaler.transform(features)
            logger.info("Transformed features with existing scaler")
        
        scaled_df = pd.DataFrame(scaled_values, columns=features.columns, index=features.index)
        return scaled_df
    
    def process_pipeline(self, tables: Dict[str, pd.DataFrame], 
                        scale: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            tables: Dictionary of table_name -> DataFrame
            scale: Whether to scale features
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("=== Starting Preprocessing Pipeline ===")
        
        # Step 1: Clean all tables
        cleaned_tables = {}
        for table_name, df in tables.items():
            cleaned_tables[table_name] = self.clean_dataframe(df, table_name)
        
        # Step 2: Merge tables
        merged_df = self.merge_tables(cleaned_tables)
        
        # Step 3: Encode categorical features
        encoded_df = self.encode_features(merged_df, fit=True)
        
        # Step 4: Prepare features and target
        features, target = self.prepare_features(encoded_df)
        
        # Step 5: Scale features (optional)
        if scale:
            features = self.scale_features(features, fit=True)
        
        logger.info("=== Preprocessing Complete ===")
        return features, target


def main():
    """Test the preprocessor."""
    from database import DatabaseConnector
    
    # Load data
    db = DatabaseConnector()
    db.connect()
    tables = db.fetch_all_tables()
    
    # Get configs
    ml_config = db.get_ml_config()
    merge_strategy = db.get_merge_strategy()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(ml_config, merge_strategy)
    
    # Run pipeline
    features, target = preprocessor.process_pipeline(tables, scale=False)
    
    print("\n=== Preprocessing Results ===")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape if target is not None else 'No target'}")
    print(f"\nFeature columns: {list(features.columns)}")
    if target is not None:
        print(f"\nTarget distribution:\n{target.value_counts()}")
    
    db.disconnect()


if __name__ == "__main__":
    main()

