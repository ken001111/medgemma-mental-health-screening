# GitHub Setup Instructions

## Quick Setup

### Option 1: Using the Setup Script (Recommended)

Run the setup script:
```bash
./setup_git.sh
```

Then follow the instructions it prints.

### Option 2: Manual Setup

1. **Initialize Git Repository** (if not already done):
   ```bash
   git init
   ```

2. **Add all files**:
   ```bash
   git add .
   ```

3. **Create initial commit**:
   ```bash
   git commit -m "Initial commit: Mental Health Screening Application with MedASR and MedGemma"
   ```

4. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Choose a repository name (e.g., `medgemma-mental-health-screening`)
   - Choose public or private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

5. **Connect local repository to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Files Included

The repository includes:
- ✅ All Python source code
- ✅ Configuration files
- ✅ Documentation (README.md, MEDGEMMA_COMPARISON.md, MEDGEMMA_USAGE.md)
- ✅ Requirements.txt
- ✅ LICENSE file
- ✅ .gitignore (excludes data, models, cache files)

## Files Excluded (via .gitignore)

- Data files (`data/`, `*.csv`, `*.db`)
- Model artifacts (`artifacts/`, `*.pkl`, `*.pt`, `*.pth`)
- Generated reports (`reports/`)
- Audio cache (`wav_cache/`, `*.wav`, `*.mp3`, `*.mp4`)
- Python cache (`__pycache__/`, `*.pyc`)
- Environment files (`.env`)

## Important Notes

1. **Sensitive Information**: The `config.py` file contains placeholder email addresses. Make sure to update these with actual values if deploying.

2. **Model Files**: Large model files (MedASR, MedGemma) are not included. Users will need to download them from HuggingFace when running the application.

3. **Data**: Training data and databases are excluded. Users should provide their own data.

## After Pushing to GitHub

1. Update the repository description on GitHub
2. Add topics/tags: `mental-health`, `medgemma`, `medasr`, `phq9`, `ptsd`, `anxiety`, `medical-ai`
3. Consider adding a GitHub Actions workflow for CI/CD (optional)
4. Add a CONTRIBUTING.md if you plan to accept contributions
