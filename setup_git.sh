#!/bin/bash
# Setup script for initializing git repository and preparing for GitHub

echo "Setting up git repository for MEDGEMMA project..."

# Remove existing git if it exists (backup first)
if [ -d ".git" ]; then
    echo "Removing existing .git directory..."
    rm -rf .git
fi

# Initialize new git repository
echo "Initializing new git repository..."
git init

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Mental Health Screening Application with MedASR and MedGemma"

echo ""
echo "✓ Git repository initialized successfully!"
echo ""
echo "Next steps to push to GitHub:"
echo "1. Create a new repository on GitHub (https://github.com/new)"
echo "2. Run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Or if you already have a GitHub repo:"
echo "   git remote add origin YOUR_GITHUB_REPO_URL"
echo "   git branch -M main"
echo "   git push -u origin main"
