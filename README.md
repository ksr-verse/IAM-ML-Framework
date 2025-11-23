# ML IAM Framework

[![License: Dual](https://img.shields.io/badge/License-Dual-blue.svg)](LICENSE)

A generic, extensible Machine Learning framework for Identity and Access Management (IAM) data analysis.

## 🎯 Overview

This framework enables IAM engineers to perform ML-based analysis on their MySQL databases (SailPoint IIQ or any relational schema) without hard-coding table structures. Simply define your tables and columns in configuration files, and the framework handles the rest.

## ✨ Key Features

- **Configuration-Driven**: Define your own tables and columns via YAML config
- **Automatic Data Processing**: Fetches, cleans, merges, and encodes data automatically
- **Multiple ML Capabilities**:
  - Classification (e.g., approval prediction)
  - Clustering (e.g., peer group discovery)
  - Regression (e.g., risk scoring)
- **Explainable Insights**: Feature importance, risk trends, access reduction opportunities
- **Visualization**: Auto-generated charts and dashboards
- **Extensible**: Easy to add new datasets, models, and analysis types

## 📁 Project Structure

```
ml-iam-framework/
├── config/
│   ├── db_config.yaml          # MySQL connection details
│   └── schema_config.yaml      # Table-column mapping definitions
├── data/
│   ├── sample_datasets/        # Dummy IAM data
│   └── generate_dummy_data.py  # Script to generate sample data
├── src/
│   ├── database.py             # Dynamic MySQL connector
│   ├── preprocessing.py        # Data cleaning and merging
│   ├── model_training.py       # Generic ML trainer
│   ├── insights.py             # Insight generator
│   └── visualization.py        # Chart generation
├── notebooks/
│   └── example_analysis.ipynb  # Interactive walkthrough
├── models/                     # Saved trained models
├── outputs/                    # Generated insights and visualizations
├── main.py                     # CLI entry point
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs all required packages including:
- Core ML libraries (scikit-learn, xgboost, lightgbm)
- Data processing (pandas, numpy)
- Database connector (mysql-connector-python)
- Visualization (matplotlib, seaborn)

### Step 2: Generate Sample Data
```bash
python data/generate_dummy_data.py
```

This generates 4 CSV files in `data/sample_datasets/`:
- `decision_history.csv` - Access request decisions
- `access_usage.csv` - Access usage patterns
- `peer_group.csv` - User organizational data
- `recertification_history.csv` - Certification history

**Optional**: Customize data generation:
```bash
python data/generate_dummy_data.py --users 1000 --access-items 200 --requests 5000
```

### Step 3: Run the Framework
```bash
python main.py --mode full
```

**That's it!** Check `outputs/` folder for results:
- `models/` - Trained ML models (.pkl files)
- `outputs/insights/` - Analysis reports (JSON/CSV)
- `outputs/visualizations/` - Charts and graphs (PNG)

---

## 🎯 Pipeline Modes

```bash
# Full pipeline (recommended for first run)
python main.py --mode full

# Training only (build models)
python main.py --mode train

# Insights only (use existing models)
python main.py --mode insights
```

## ⚙️ Configuration

### For Sample Data (Default - Ready to Use)
Already configured in `config/db_config.yaml`:
```yaml
use_sample_data: true
```

The framework includes 4 pre-configured tables:
- **decision_history**: Access request approval/rejection decisions
- **access_usage**: How frequently access items are used
- **peer_group**: User organizational context (role, department, location)
- **recertification_history**: Past certification decisions

### For MySQL Database
Edit `config/db_config.yaml`:
```yaml
use_sample_data: false

mysql:
  host: <your_mysql_host>
  port: <your_mysql_port>
  database: <your_database_name>
  user: <your_username>
  password: <your_password>
  # Optional: Enhanced connection parameters
  connection_timeout: 10  # Timeout in seconds
  charset: utf8mb4        # Character encoding
  autocommit: true        # Auto-commit transactions
  use_unicode: true       # Unicode string handling
  raise_on_warnings: false
```

### Define Your Tables
Edit `config/schema_config.yaml`:
```yaml
tables:
  decision_history:
    columns: [request_id, user_id, access_item, decision, timestamp, requester_role, approver_id, risk_score, sod_conflict, is_privileged_access]
    join_keys: [user_id, access_item]
  access_usage:
    columns: [user_id, access_item, last_used_date, frequency, access_type]
    join_keys: [user_id, access_item]
  peer_group:
    columns: [user_id, peer_group_id, role, department, seniority_level, location, employment_type]
    join_keys: [user_id]
  recertification_history:
    columns: [user_id, access_item, last_certified_date, certification_outcome, certified_by]
    join_keys: [user_id, access_item]
```

### ML Algorithm Configuration
The framework supports multiple ML algorithms with configurable priority order:
```yaml
ml_config:
  models:
    classification:
      enabled: true
      # Priority order: RF → GB → LightGBM → XGBoost → Logistic
      algorithms: [random_forest, gradient_boosting, lightgbm, xgboost, logistic_regression]
```

**Available Algorithms:**
- **Random Forest (RF)**: Fast, interpretable, good baseline
- **Gradient Boosting (GB)**: Strong performance, feature importance
- **LightGBM**: Fast gradient boosting, handles large datasets
- **XGBoost**: High performance, robust to overfitting
- **Logistic Regression**: Fast, interpretable, good baseline

## 📊 Data Structure

The framework works with 4 core IAM tables:

1. **decision_history**: Access request decisions
   - Tracks approval/rejection decisions with risk scores
   - Includes SoD conflicts and privileged access flags
   - Primary key: `request_id`

2. **access_usage**: Access usage patterns
   - Frequency of access usage
   - Last used dates
   - Access types (Role, Entitlement, etc.)

3. **peer_group**: Organizational context
   - User roles, departments, locations
   - Seniority levels and employment types
   - Peer group assignments

4. **recertification_history**: Certification records
   - Past certification decisions
   - Certification outcomes (approved/revoked/deferred)
   - Certification dates and certifiers

All tables are automatically merged using join keys defined in `schema_config.yaml`.

## 📊 Use Cases

### 1. Approval Prediction
Predict whether access requests will be approved or rejected based on historical patterns, risk scores, and user context.

### 2. Risk Scoring
Identify high-risk access patterns and users requiring attention using ML-based risk assessment.

### 3. Access Reduction
Find unused or low-frequency entitlements that can be safely removed to reduce access sprawl.

### 4. Peer Group Analysis
Discover natural user clusters for role-based access recommendations using clustering algorithms.

### 5. Anomaly Detection
Identify unusual access patterns or privilege creep through ML-based anomaly detection.

## 🔧 Customization

### Adding New Tables

Simply edit `config/schema_config.yaml`:

```yaml
tables:
  your_new_table:
    columns: [col1, col2, col3]
    join_keys: [user_id]  # Optional: for auto-merging
```

### Adding New Models

Extend `src/model_training.py` with your own sklearn or custom models.

### Custom Insights

Add analysis functions to `src/insights.py` to generate domain-specific insights.

## 📈 Output Examples

- **Models**: Saved in `models/` directory (`.pkl` files)
- **Insights**: JSON and CSV reports in `outputs/insights/`
- **Visualizations**: PNG charts in `outputs/visualizations/`

## 🤝 Contributing

This is an extensible framework. Feel free to:
- Add new preprocessing techniques
- Integrate deep learning models
- Add real-time prediction APIs
- Create custom dashboards

## 📝 License

This project is licensed under a **Dual License**:
- **Free** for personal and educational use
- **Commercial license required** for company/organizational use

See the [LICENSE](LICENSE) file for complete terms and contact information.

## 🛠️ Requirements

- Python 3.8+
- MySQL database (or sample CSV data)
- See `requirements.txt` for Python packages

## 📚 Documentation

For detailed examples and walkthroughs, see:
- `notebooks/example_analysis.ipynb` - Interactive tutorial
- Comments in source code for implementation details

## 👥 For Developers

### Project Structure Details
- **`config/`**: Configuration files (YAML format)
  - `db_config.yaml`: Database connection settings
  - `schema_config.yaml`: Table definitions and ML configuration
- **`data/`**: Data files and generators
  - `sample_datasets/`: CSV data files (auto-generated)
  - `generate_dummy_data.py`: Script to generate sample data
- **`src/`**: Core framework modules
  - `database.py`: Handles MySQL/CSV data loading with automatic path resolution
  - `preprocessing.py`: Data cleaning, merging, and feature engineering
  - `model_training.py`: ML model training with configurable algorithms
  - `insights.py`: Insight generation and analysis
  - `visualization.py`: Chart and graph generation

### Key Features for Developers
1. **Automatic Path Resolution**: Works from any directory, resolves paths relative to project root
2. **Configuration-Driven**: All settings in YAML files, no code changes needed
3. **Extensible Architecture**: Easy to add new models, insights, or preprocessing steps
4. **Robust Error Handling**: Comprehensive logging and graceful error handling
5. **Algorithm Priority**: Respects algorithm order from config file

### Adding New Features
- **New Tables**: Add to `config/schema_config.yaml` under `tables:`
- **New Models**: Extend `src/model_training.py` and add to config `algorithms:`
- **New Insights**: Add functions to `src/insights.py` and reference in config
- **New Visualizations**: Add methods to `src/visualization.py`

## 🐛 Troubleshooting

### Connection Issues
- **MySQL Connection Failures**: 
  - Verify credentials in `config/db_config.yaml`
  - Check if MySQL server is running and accessible
  - Verify network connectivity and firewall settings
  - Try increasing `connection_timeout` value
  - Ensure database exists and user has proper permissions

- **CSV File Not Found**:
  - Run `python data/generate_dummy_data.py` to generate sample data
  - Verify files exist in `data/sample_datasets/` directory
  - Check that `use_sample_data: true` in `config/db_config.yaml`

### Data Loading Issues
- **Missing Columns**: 
  - Ensure `schema_config.yaml` matches your actual database schema
  - Verify all required columns are listed in the table definitions
  - Check column names match exactly (case-sensitive)

- **Empty DataFrames**:
  - Verify your database tables contain data
  - Check that table names in config match database table names
  - Review log output for specific error messages

### ML Training Issues
- **Algorithm Not Available**:
  - LightGBM/XGBoost: Run `pip install lightgbm xgboost`
  - Check logs for missing library warnings
  - Algorithms will be skipped if not installed (with warning)

- **Memory Errors**:
  - Reduce dataset size in data generator
  - Use `--mode train` to train models only (no visualizations)
  - Consider sampling data for large datasets

### Import Errors
- **ModuleNotFoundError**: 
  - Run `pip install -r requirements.txt` to install all dependencies
  - Ensure you're using Python 3.8 or higher
  - Consider using a virtual environment

- **Path Resolution Issues**:
  - Always run commands from the project root directory
  - The framework automatically resolves paths relative to project root
  - Check log output for resolved paths if issues occur

### General Tips
- Check `iam_ml_framework.log` for detailed error messages
- Run with `--mode train` first to test data loading and preprocessing
- Verify all 4 tables are loaded successfully before training
- Ensure sufficient disk space for model files and outputs

---

**Happy IAM Analysis! 🔐📊**

