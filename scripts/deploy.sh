#!/bin/bash

set -e

# Where to deploy in WSL
DEPLOY_DIR=~/ticket_deploy

echo "🚀 Deploying to $DEPLOY_DIR"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy models, data, and outputs (adjust paths if needed)
cp -r models "$DEPLOY_DIR/"
cp -r data "$DEPLOY_DIR/"
cp -r outputs "$DEPLOY_DIR/"

echo "✅ Deployment completed at $DEPLOY_DIR"
