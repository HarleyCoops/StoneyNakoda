# Setup Bak-Sneppen Repository
# ==============================
# This script creates a clean repository with only Bak-Sneppen files

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         BAK-SNEPPEN REPOSITORY SETUP                       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Define the new repository directory
$newRepoPath = "..\Bak-Sneppen"
$currentPath = Get-Location

Write-Host "Creating clean Bak-Sneppen repository at: $newRepoPath`n" -ForegroundColor Yellow

# Create new directory
if (Test-Path $newRepoPath) {
    Write-Host "Directory already exists. Removing old version..." -ForegroundColor Yellow
    Remove-Item -Path $newRepoPath -Recurse -Force
}

New-Item -Path $newRepoPath -ItemType Directory | Out-Null
Write-Host "✓ Created directory: $newRepoPath" -ForegroundColor Green

# List of Bak-Sneppen files to copy
$filesToCopy = @(
    # Python source files
    "bak_sneppen_3d.py",
    "bak_sneppen_3d_enhanced.py",
    "bak_sneppen_config.py",
    "bak_sneppen_interactive.py",
    "analyze_bak_sneppen_data.py",
    
    # PowerShell scripts
    "run_bak_sneppen.ps1",
    "demo_all_features.ps1",
    "setup_bak_sneppen_repo.ps1",
    
    # Documentation
    "INDEX.md",
    "QUICK_REFERENCE.md",
    "README_BAK_SNEPPEN_COMPLETE.md",
    "BAK_SNEPPEN_README.md",
    "MATH_TO_MANIM_PATTERNS.md",
    "PROJECT_SUMMARY.md",
    "VISUAL_OVERVIEW.md",
    
    # Requirements
    "requirements_bak_sneppen.txt"
)

Write-Host "`nCopying files..." -ForegroundColor Yellow

$copiedCount = 0
$missingFiles = @()

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Copy-Item -Path $file -Destination $newRepoPath -Force
        Write-Host "  ✓ $file" -ForegroundColor Gray
        $copiedCount++
    } else {
        Write-Host "  ✗ $file (not found)" -ForegroundColor Red
        $missingFiles += $file
    }
}

Write-Host "`n✓ Copied $copiedCount files" -ForegroundColor Green

if ($missingFiles.Count -gt 0) {
    Write-Host "`n⚠ Missing files: $($missingFiles.Count)" -ForegroundColor Yellow
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Yellow
    }
}

# Create README.md for the new repo
Write-Host "`nCreating main README.md..." -ForegroundColor Yellow

$readmeContent = @"
# 🌀 Bak-Sneppen 3D Visualization Suite

**An epic, rotating 3D visualization of evolutionary self-organized criticality**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Manim](https://img.shields.io/badge/Manim-Community-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Quick Start

``````powershell
# Install dependencies
pip install manim numpy matplotlib

# Render the epic 3D visualization
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Or try interactive mode
python bak_sneppen_interactive.py basic
``````

---

## 📚 What's Inside

This comprehensive suite provides **multiple ways** to explore the Bak-Sneppen model—a canonical example of self-organized criticality:

### 🎬 Visualizations
- **3D Rotating Animations** - Cinematic Manim scenes with smooth camera work
- **Interactive Simulator** - Real-time matplotlib exploration with manual stepping
- **Multiple Scene Types** - Main evolution, histogram, avalanche close-ups

### ⚙️ Features
- **Configuration System** - 7 ready-to-use presets for different scenarios
- **Data Export & Analysis** - JSON export with complete analysis pipeline
- **PowerShell Automation** - One-command rendering on Windows
- **Cross-Platform** - Works on Windows, macOS, Linux

### 📊 What You Get
- **MP4 Videos** - From 480p to 4K quality
- **Statistical Plots** - Fitness evolution, distributions, avalanches
- **Data Files** - Complete simulation history in JSON
- **Analysis Reports** - Summary statistics and metrics

---

## 🎯 The Bak-Sneppen Model

The Bak-Sneppen model demonstrates **self-organized criticality** in evolutionary systems:

1. **Species Circle** - N species arranged in a circle
2. **Random Fitness** - Each has fitness value [0, 1]
3. **Evolution Rule** - Replace weakest + neighbors with new random values
4. **Emergent Behavior** - Cascading avalanches and power law distributions

**Key Insight**: The system naturally evolves to a critical state (~0.667 fitness threshold) without external tuning.

---

## 📖 Documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page command cheatsheet
- **[README_BAK_SNEPPEN_COMPLETE.md](README_BAK_SNEPPEN_COMPLETE.md)** - Complete user guide
- **[MATH_TO_MANIM_PATTERNS.md](MATH_TO_MANIM_PATTERNS.md)** - Design patterns explained
- **[INDEX.md](INDEX.md)** - Navigate all files

---

## 🎨 Example Outputs

### Main 3D Visualization
- Rotating camera with 3D species arrangement
- Color-coded fitness (red → yellow → green)
- Real-time fitness distribution graph
- Animated avalanche cascades
- 2-3 minute videos

### Interactive Dashboard
- 6-panel live visualization
- Manual stepping with SPACE key
- Perfect for teaching and exploration
- No rendering wait time

---

## 🔬 Scientific Context

**Paper**: Bak, P., & Sneppen, K. (1993). "Punctuated equilibrium and criticality in a simple model of evolution." *Physical Review Letters*, 71(24), 4083.

**Applications**:
- Evolution & ecology (punctuated equilibrium)
- Economics (market crashes)
- Neuroscience (neural avalanches)
- Seismology (earthquake distributions)

---

## 🛠️ Usage Examples

### Render High-Quality Video
``````powershell
# Use PowerShell launcher
.\run_bak_sneppen.ps1 -Scene basic -Quality high

# Or direct manim command
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D
``````

### Interactive Exploration
``````powershell
# Automated demo
python bak_sneppen_interactive.py basic

# Manual stepping (SPACE to advance)
python bak_sneppen_interactive.py manual

# Compare system sizes
python bak_sneppen_interactive.py compare
``````

### Customize Parameters
``````python
# Edit bak_sneppen_config.py
ACTIVE_CONFIG = ArtisticConfig  # Try different presets!
``````

### Analyze Data
``````powershell
# Render enhanced scene (exports JSON)
manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced

# Analyze the data
python analyze_bak_sneppen_data.py --latest
``````

---

## 📦 Project Structure

``````
Bak-Sneppen/
├── Source Files
│   ├── bak_sneppen_3d.py              # Main 3D scenes
│   ├── bak_sneppen_3d_enhanced.py     # Enhanced + data export
│   ├── bak_sneppen_config.py          # Configuration system
│   ├── bak_sneppen_interactive.py     # Interactive simulator
│   └── analyze_bak_sneppen_data.py    # Analysis pipeline
│
├── Scripts
│   ├── run_bak_sneppen.ps1            # PowerShell launcher
│   └── demo_all_features.ps1          # Feature walkthrough
│
├── Documentation
│   ├── README.md                       # This file
│   ├── QUICK_REFERENCE.md             # Command cheatsheet
│   ├── README_BAK_SNEPPEN_COMPLETE.md # Complete guide
│   ├── MATH_TO_MANIM_PATTERNS.md      # Design patterns
│   └── INDEX.md                        # File navigation
│
└── requirements_bak_sneppen.txt       # Python dependencies
``````

---

## 🎓 Educational Use

Perfect for:
- **University Courses** - Physics, biology, complex systems
- **Research Presentations** - Conferences and seminars
- **Public Science Communication** - YouTube, festivals, museums
- **Self-Study** - Interactive learning of complex systems

---

## 🙏 Acknowledgments

- **Per Bak & Kim Sneppen** - Original model creators (1993)
- **Grant Sanderson (3Blue1Brown)** - Manim creator
- **Manim Community** - Maintaining Manim Community Edition
- **Math-To-Manim Project** - Design pattern inspiration

---

## 📝 License

This project is provided for educational and research purposes. When using:
- Cite the original Bak-Sneppen model (Bak & Sneppen, 1993)
- Credit Manim Community Edition
- Share your amazing visualizations!

---

## 🌟 Features Highlight

- ✨ **15 Design Patterns** from Math-To-Manim
- 🎬 **4 Scene Types** for different perspectives
- ⚙️ **7 Configuration Presets** for easy customization
- 📊 **4 Analysis Plot Types** for deep insights
- 🎮 **Multiple Interaction Modes** (render/explore/analyze/teach)
- 📚 **3,100+ Lines of Documentation** - More docs than code!
- 🚀 **One-Command Rendering** via PowerShell launcher
- 🔬 **Research-Grade Output** with data export

---

## 💫 Get Started in 3 Minutes

``````powershell
# 1. Install (30 seconds)
pip install manim numpy matplotlib

# 2. Render (2 minutes)
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D

# 3. Watch and enjoy! 🎉
``````

---

**Visualizing complexity, one beautiful frame at a time.** 🌀✨

For questions or issues, see the [complete documentation](README_BAK_SNEPPEN_COMPLETE.md) or check the [quick reference](QUICK_REFERENCE.md).
"@

Set-Content -Path "$newRepoPath\README.md" -Value $readmeContent
Write-Host "✓ Created README.md" -ForegroundColor Green

# Create .gitignore
Write-Host "`nCreating .gitignore..." -ForegroundColor Yellow

$gitignoreContent = @"
# Manim output directories
media/
__pycache__/
*.pyc
*.pyo

# Data files (generated)
bak_sneppen_data_*.json
bak_sneppen_data_*.png
bak_sneppen_data_*_report.txt

# Python virtual environment
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# LaTeX auxiliary files (if using LaTeX rendering)
*.aux
*.log
*.out
*.toc

# Temporary files
*.tmp
*.bak
"@

Set-Content -Path "$newRepoPath\.gitignore" -Value $gitignoreContent
Write-Host "✓ Created .gitignore" -ForegroundColor Green

# Initialize git repository
Write-Host "`nInitializing Git repository..." -ForegroundColor Yellow
Set-Location $newRepoPath

git init | Out-Null
Write-Host "✓ Git repository initialized" -ForegroundColor Green

# Add all files
Write-Host "`nStaging files..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged" -ForegroundColor Green

# Create initial commit
Write-Host "`nCreating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Bak-Sneppen 3D Visualization Suite

- Epic rotating 3D visualization with Manim
- Interactive real-time simulator
- Configuration system with 7 presets
- Data export and analysis pipeline
- PowerShell automation scripts
- Comprehensive documentation (3,100+ lines)
- Math-To-Manim design patterns
- Research-grade output quality

Features:
- 4 scene types (main, enhanced, histogram, avalanche)
- Multiple interaction modes (render/explore/analyze/teach)
- Cross-platform support (Windows/macOS/Linux)
- Quick start in 3 minutes

Inspired by Math-To-Manim patterns for professional mathematical animation."

Write-Host "✓ Initial commit created" -ForegroundColor Green

# Add remote
Write-Host "`nAdding remote repository..." -ForegroundColor Yellow
git remote add origin https://github.com/HarleyCoops/Bak-Sneppen.git
Write-Host "✓ Remote added: https://github.com/HarleyCoops/Bak-Sneppen.git" -ForegroundColor Green

# Create main branch
Write-Host "`nSetting up main branch..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Branch renamed to 'main'" -ForegroundColor Green

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    SETUP COMPLETE!                         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "Repository ready at: $newRepoPath`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd $newRepoPath" -ForegroundColor White
Write-Host "  2. git push -u origin main" -ForegroundColor White
Write-Host "`n  Or create a feature branch for a PR:" -ForegroundColor Gray
Write-Host "  3. git checkout -b feature/initial-implementation" -ForegroundColor White
Write-Host "  4. git push -u origin feature/initial-implementation" -ForegroundColor White
Write-Host "  5. Open PR on GitHub`n" -ForegroundColor White

Write-Host "Repository structure:" -ForegroundColor Yellow
git log --oneline
Write-Host ""
git status

Set-Location $currentPath
Write-Host "`nReturned to: $currentPath`n" -ForegroundColor Gray


