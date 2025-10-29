# Push Bak-Sneppen Files to New Repository

## Step-by-Step Instructions

### 1. Create New Directory

```powershell
# Navigate to parent directory
cd ..

# Create new directory for Bak-Sneppen repo
mkdir Bak-Sneppen
cd Bak-Sneppen
```

### 2. Copy These Files

Copy the following files from `StoneyNakoda` to `Bak-Sneppen` directory:

**Python Source Files:**
- `bak_sneppen_3d.py`
- `bak_sneppen_3d_enhanced.py`
- `bak_sneppen_config.py`
- `bak_sneppen_interactive.py`
- `analyze_bak_sneppen_data.py`

**PowerShell Scripts:**
- `run_bak_sneppen.ps1`
- `demo_all_features.ps1`

**Documentation:**
- `INDEX.md`
- `QUICK_REFERENCE.md`
- `README_BAK_SNEPPEN_COMPLETE.md`
- `BAK_SNEPPEN_README.md`
- `MATH_TO_MANIM_PATTERNS.md`
- `PROJECT_SUMMARY.md`
- `VISUAL_OVERVIEW.md`

**Requirements:**
- `requirements_bak_sneppen.txt`

### 3. Quick Copy Command

```powershell
# From the Bak-Sneppen directory, run:
Copy-Item ..\StoneyNakoda\bak_sneppen_*.py .
Copy-Item ..\StoneyNakoda\analyze_bak_sneppen_data.py .
Copy-Item ..\StoneyNakoda\run_bak_sneppen.ps1 .
Copy-Item ..\StoneyNakoda\demo_all_features.ps1 .
Copy-Item ..\StoneyNakoda\*BAK_SNEPPEN*.md .
Copy-Item ..\StoneyNakoda\INDEX.md .
Copy-Item ..\StoneyNakoda\QUICK_REFERENCE.md .
Copy-Item ..\StoneyNakoda\MATH_TO_MANIM_PATTERNS.md .
Copy-Item ..\StoneyNakoda\PROJECT_SUMMARY.md .
Copy-Item ..\StoneyNakoda\VISUAL_OVERVIEW.md .
Copy-Item ..\StoneyNakoda\requirements_bak_sneppen.txt .
```

### 4. Initialize Git Repository

```powershell
# Initialize git
git init

# Create main README.md (see below)
# Create .gitignore (see below)

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Bak-Sneppen 3D Visualization Suite

- Epic rotating 3D visualization with Manim
- Interactive real-time simulator
- Configuration system with 7 presets
- Data export and analysis pipeline
- Comprehensive documentation
- Math-To-Manim design patterns"
```

### 5. Connect to GitHub and Push

```powershell
# Add remote
git remote add origin https://github.com/HarleyCoops/Bak-Sneppen.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

### 6. (Optional) Create Pull Request Branch

```powershell
# Create feature branch
git checkout -b feature/initial-implementation

# Push feature branch
git push -u origin feature/initial-implementation

# Then create PR on GitHub web interface
```

---

## Files to Create

### README.md

Create a file called `README.md` in the Bak-Sneppen directory with the main project overview. (Content provided separately)

### .gitignore

```
# Manim output
media/
__pycache__/
*.pyc

# Generated data
bak_sneppen_data_*.json
bak_sneppen_data_*.png
bak_sneppen_data_*.txt

# Python
venv/
.venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## Complete Command Sequence

```powershell
# 1. Go to parent directory
cd ..

# 2. Create and enter new directory
mkdir Bak-Sneppen
cd Bak-Sneppen

# 3. Copy all Bak-Sneppen files
Copy-Item ..\StoneyNakoda\bak_sneppen_*.py .
Copy-Item ..\StoneyNakoda\analyze_bak_sneppen_data.py .
Copy-Item ..\StoneyNakoda\run_bak_sneppen.ps1 .
Copy-Item ..\StoneyNakoda\demo_all_features.ps1 .
Copy-Item ..\StoneyNakoda\*BAK_SNEPPEN*.md .
Copy-Item ..\StoneyNakoda\INDEX.md .
Copy-Item ..\StoneyNakoda\QUICK_REFERENCE.md .
Copy-Item ..\StoneyNakoda\MATH_TO_MANIM_PATTERNS.md .
Copy-Item ..\StoneyNakoda\PROJECT_SUMMARY.md .
Copy-Item ..\StoneyNakoda\VISUAL_OVERVIEW.md .
Copy-Item ..\StoneyNakoda\requirements_bak_sneppen.txt .

# 4. Create README.md and .gitignore (manually or see templates)

# 5. Initialize git and commit
git init
git add .
git commit -m "Initial commit: Bak-Sneppen 3D Visualization Suite"

# 6. Connect to GitHub
git remote add origin https://github.com/HarleyCoops/Bak-Sneppen.git
git branch -M main

# 7. Push to GitHub
git push -u origin main
```

---

## Verify Before Pushing

```powershell
# Check what will be committed
git status

# View commit history
git log --oneline

# List files
dir
```

---

## Total Files: 17

- 5 Python files
- 2 PowerShell scripts
- 8 Markdown documentation files
- 1 Requirements file
- 1 README.md (to create)
- 1 .gitignore (to create)

