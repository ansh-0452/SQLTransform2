# SQLTransformer: Natural Language to SQL Translation Engine

An open-source, local transformer architecture that converts plain English questions into production SQL queries without external AI APIs or cloud dependencies.

## Quickstart
```bash
# 1. Run local launcher with one command
chmod +x run.sh
./run.sh

# 2. Or run Python CLI directly:
python3 transformer.py --query "Show top 5 customers ordered by total order amount" --dialect postgresql
```

## Architecture
- Standalone `transformer.py` with Multi-Head Attention, Positional Encoding, and Entity Linker.
- Multi-dialect SQL generation (PostgreSQL, MySQL, SQLite, BigQuery, Oracle).
- Automated test suite in `tests/test_transformer.py`.
