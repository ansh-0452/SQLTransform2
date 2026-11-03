#!/usr/bin/env bash
set -e
echo "=========================================================="
echo " Starting SQLTransformer - Local Natural Language to SQL"
echo "=========================================================="
python3 transformer.py --query "Find all customers with orders over 500" --dialect postgresql
echo ""
echo "[*] Starting dev server..."
npm run dev
