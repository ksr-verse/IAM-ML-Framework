"""
Database Module - Dynamic Data Fetching
Reads MySQL connection info and schema details from config files
Fetches data from specified tables dynamically
Returns pandas DataFrames for each defined table
"""

import os
import yaml
import pandas as pd
import mysql.connector
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Generic database connector that works with MySQL or CSV files.
    Configuration-driven - reads table definitions from YAML.
    """
    
    def __init__(self, db_config_path: str = 'config/db_config.yaml',
                 schema_config_path: str = 'config/schema_config.yaml'):
        """
        Initialize database connector with configuration files.
        
        Args:
            db_config_path: Path to database configuration file
            schema_config_path: Path to schema configuration file
        """
        # Resolve paths to absolute paths
        self.db_config_path = self._resolve_path(db_config_path)
        self.schema_config_path = self._resolve_path(schema_config_path)
        
        # Determine project root (parent of config directory)
        if 'config' in os.path.dirname(self.db_config_path):
            self.project_root = os.path.dirname(os.path.dirname(self.db_config_path))
        else:
            # Fallback: use current working directory
            self.project_root = os.getcwd()
        
        self.db_config = self._load_config(self.db_config_path)
        self.schema_config = self._load_config(self.schema_config_path)
        self.connection = None
        self.use_sample_data = self.db_config.get('use_sample_data', True)
        
        logger.info(f"Project root directory: {self.project_root}")
        if self.use_sample_data:
            logger.info("Mode: Using sample CSV data files")
        else:
            logger.info("Mode: Using MySQL database connection")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path to absolute path."""
        if os.path.isabs(path):
            return path
        # Try relative to current working directory first
        if os.path.exists(path):
            return os.path.abspath(path)
        # Try relative to this file's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # Go up from src/ to project root
        resolved = os.path.join(parent_dir, path)
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
        # Return as-is if not found (will raise error in _load_config)
        return os.path.abspath(path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def connect(self) -> bool:
        """
        Establish connection to MySQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.use_sample_data:
            logger.info("Using sample CSV data instead of MySQL")
            return True
            
        try:
            mysql_config = self.db_config['mysql']
            
            # Build connection parameters with defaults
            connection_params = {
                'host': mysql_config['host'],
                'port': mysql_config['port'],
                'database': mysql_config['database'],
                'user': mysql_config['user'],
                'password': mysql_config['password']
            }
            
            # Add optional connection parameters if specified
            optional_params = [
                'connection_timeout', 'charset', 'autocommit', 
                'use_unicode', 'raise_on_warnings'
            ]
            for param in optional_params:
                if param in mysql_config:
                    connection_params[param] = mysql_config[param]
            
            self.connection = mysql.connector.connect(**connection_params)
            logger.info(f"Connected to MySQL database: {mysql_config['database']}")
            return True
        except mysql.connector.Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Disconnected from MySQL database")
    
    def fetch_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from a specified table.
        
        Args:
            table_name: Name of the table to fetch
            
        Returns:
            pandas DataFrame with the table data, or None if error
        """
        if self.use_sample_data:
            return self._fetch_from_csv(table_name)
        else:
            return self._fetch_from_mysql(table_name)
    
    def _fetch_from_csv(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from CSV file.
        
        Args:
            table_name: Name of the table (matches CSV filename)
            
        Returns:
            pandas DataFrame with the table data, or None if error
        """
        try:
            sample_config = self.db_config['sample_data']
            data_dir = sample_config['directory']
            filename = sample_config['files'].get(table_name, f"{table_name}.csv")
            
            # Resolve path relative to project root
            if not os.path.isabs(data_dir):
                filepath = os.path.join(self.project_root, data_dir, filename)
            else:
                filepath = os.path.join(data_dir, filename)
            
            # Normalize path (handle .. and .)
            filepath = os.path.normpath(filepath)
            
            if not os.path.exists(filepath):
                logger.warning(f"CSV file not found: {filepath}")
                logger.warning(f"  Project root: {self.project_root}")
                logger.warning(f"  Looking for: {table_name}")
                return None
                
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from CSV: {os.path.basename(filepath)}")
            logger.debug(f"Full path: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV for table {table_name}: {e}")
            return None
    
    def _fetch_from_mysql(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from MySQL table.
        
        Args:
            table_name: Name of the table to fetch
            
        Returns:
            pandas DataFrame with the table data, or None if error
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected to database. Call connect() first.")
            return None
            
        try:
            # Get column definitions from schema config
            table_config = self.schema_config['tables'].get(table_name)
            if not table_config:
                logger.warning(f"Table {table_name} not found in schema config")
                # Fetch all columns
                query = f"SELECT * FROM {table_name}"
            else:
                columns = table_config['columns']
                columns_str = ', '.join(columns)
                query = f"SELECT {columns_str} FROM {table_name}"
            
            df = pd.read_sql(query, self.connection)
            logger.info(f"Fetched {len(df)} rows from MySQL table: {table_name}")
            return df
            
        except mysql.connector.Error as e:
            logger.error(f"Error fetching table {table_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching table {table_name}: {e}")
            return None
    
    def fetch_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all tables defined in schema configuration.
        
        Returns:
            Dictionary mapping table names to DataFrames
        """
        tables = {}
        table_names = self.schema_config['tables'].keys()
        
        logger.info(f"Fetching {len(table_names)} tables: {list(table_names)}")
        
        for table_name in table_names:
            df = self.fetch_table(table_name)
            if df is not None:
                tables[table_name] = df
            else:
                logger.warning(f"Skipping table {table_name} due to errors")
        
        logger.info(f"Successfully fetched {len(tables)} tables")
        return tables
    
    def get_table_config(self, table_name: str) -> Optional[dict]:
        """
        Get configuration for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table configuration
        """
        return self.schema_config['tables'].get(table_name)
    
    def get_merge_strategy(self) -> Optional[dict]:
        """
        Get the merge strategy from configuration.
        
        Returns:
            Dictionary with merge strategy configuration
        """
        return self.schema_config.get('merge_strategy')
    
    def get_ml_config(self) -> Optional[dict]:
        """
        Get ML configuration from schema config.
        
        Returns:
            Dictionary with ML configuration
        """
        return self.schema_config.get('ml_config')
    
    def get_insights_config(self) -> Optional[dict]:
        """
        Get insights configuration from schema config.
        
        Returns:
            Dictionary with insights configuration
        """
        return self.schema_config.get('insights_config')


def main():
    """Test the database connector."""
    # Initialize connector
    db = DatabaseConnector()
    
    # Connect to database (or use sample data)
    if db.connect():
        # Fetch all tables
        tables = db.fetch_all_tables()
        
        # Print summary
        print("\n=== Database Summary ===")
        for table_name, df in tables.items():
            print(f"\nTable: {table_name}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample:\n{df.head(3)}")
        
        # Disconnect
        db.disconnect()
    else:
        print("Failed to connect to database")


if __name__ == "__main__":
    main()

