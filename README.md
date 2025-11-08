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

### Step 2: Generate Sample Data
```bash
python data/generate_dummy_data.py
```

### Step 3: Run the Framework
```bash
python main.py --mode full
```

**That's it!** Check `outputs/` folder for results.

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
```

### Define Your Tables
Edit `config/schema_config.yaml`:
```yaml
tables:
  decision_history:
    columns: [request_id, user_id, access_item, decision, timestamp]
  access_usage:
    columns: [user_id, access_item, last_used_date, frequency]
```

## 📊 Use Cases

### 1. Approval Prediction
Predict whether access requests will be approved or rejected based on historical patterns.

### 2. Risk Scoring
Identify high-risk access patterns and users requiring attention.

### 3. Access Reduction
Find unused or low-frequency entitlements that can be safely removed.

### 4. Peer Group Analysis
Discover natural user clusters for role-based access recommendations.

### 5. Anomaly Detection
Identify unusual access patterns or privilege creep.

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

## 🐛 Troubleshooting

**Connection Issues**: Verify MySQL credentials in `config/db_config.yaml`

**Missing Columns**: Ensure `schema_config.yaml` matches your actual database schema

**Import Errors**: Run `pip install -r requirements.txt` to install all dependencies

---

**Happy IAM Analysis! 🔐📊**

