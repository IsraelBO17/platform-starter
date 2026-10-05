#!/bin/bash
# build_lambda.sh - build function.zip for the Lambda console (Python 3.12, x86_64).
set -euo pipefail
rm -rf build function.zip
pip install -r requirements.txt --target build \
  --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --only-binary=:all:
cp -r app build/
cp run.sh build/ && chmod +x build/run.sh
(cd build && zip -qr ../function.zip . -x "*/__pycache__/*")
echo "Built function.zip ($(du -h function.zip | cut -f1))"
