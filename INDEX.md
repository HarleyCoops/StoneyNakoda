# 📚 Bak-Sneppen 3D Visualization - File Index

## Quick Navigation

**→ New User?** Start with [`QUICK_REFERENCE.md`](#quick_referencemd)  
**→ Want Full Docs?** Read [`README_BAK_SNEPPEN_COMPLETE.md`](#readme_bak_sneppen_completemd)  
**→ Ready to Render?** Run `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D`  
**→ Want to Explore?** Try `python bak_sneppen_interactive.py basic`

---

## 📂 File Organization

### 🎬 Visualization Files (Run These)

#### `bak_sneppen_3d.py`
**What**: Main 3D visualization scenes  
**Contains**: 4 scene classes (Evolution3D, Histogram, Avalanche, and base)  
**Lines**: 433  
**Run**: `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D`  
**Best For**: Standard high-quality 3D rotating visualization

#### `bak_sneppen_3d_enhanced.py`
**What**: Enhanced version with data tracking and export  
**Contains**: BakSneppenEnhanced scene with JSON export  
**Lines**: 390  
**Run**: `manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced`  
**Best For**: Research use, when you need data for analysis  
**Output**: MP4 video + JSON data file

#### `bak_sneppen_interactive.py`
**What**: Real-time matplotlib interactive simulator  
**Contains**: 6-panel dashboard with live updates  
**Lines**: 495  
**Run**: `python bak_sneppen_interactive.py basic`  
**Modes**: `basic`, `large`, `manual`, `compare`  
**Best For**: Exploration, teaching, quick experiments

---

### ⚙️ Configuration & Analysis

#### `bak_sneppen_config.py`
**What**: Centralized configuration system  
**Contains**: 7 preset configurations + base config class  
**Lines**: 448  
**Edit**: Change `ACTIVE_CONFIG = ...` to switch presets  
**Presets**:
- `BakSneppenConfig` - Default (30 species, 50 iter)
- `QuickPreviewConfig` - Fast (15 species, 20 iter)
- `DetailedConfig` - High quality (50 species, 100 iter)
- `LargeScaleConfig` - Large (100 species, 200 iter)
- `ArtisticConfig` - Purple→Pink→Orange colors
- `MinimalistConfig` - Clean minimal style
- `EducationalConfig` - Slow, labeled

#### `analyze_bak_sneppen_data.py`
**What**: Data analysis and plotting pipeline  
**Contains**: Complete analysis suite with 4 plot types  
**Lines**: 433  
**Run**: `python analyze_bak_sneppen_data.py --latest`  
**Input**: JSON files from enhanced scene  
**Output**: PNG plots + text report  
**Best For**: Post-simulation statistical analysis

---

### 🚀 Automation Scripts

#### `run_bak_sneppen.ps1`
**What**: PowerShell launcher for Windows  
**Contains**: One-command rendering with options  
**Lines**: 98  
**Run**: `.\run_bak_sneppen.ps1 -Scene basic -Quality high`  
**Options**: 
- `-Scene`: basic, enhanced, histogram, avalanche, all
- `-Quality`: low, medium, high, 4k
- `-NoPreview`: Don't play after rendering
- `-ExportGif`: Export as GIF instead of MP4

#### `demo_all_features.ps1`
**What**: Complete feature demonstration script  
**Contains**: Interactive walkthrough of all features  
**Lines**: 300+  
**Run**: `.\demo_all_features.ps1`  
**Best For**: First-time setup, seeing all capabilities

---

### 📚 Documentation Files (Read These)

#### `QUICK_REFERENCE.md`
**What**: One-page quick reference card  
**Lines**: 194  
**Contents**: 
- One-line commands
- Quality flags table
- Configuration quick edit
- Common customizations
- Troubleshooting table
**Best For**: Quick lookup, daily use

#### `README_BAK_SNEPPEN_COMPLETE.md`
**What**: Complete user guide and manual  
**Lines**: 728  
**Contents**:
- Overview and quick start
- All features documented
- Usage scenarios and workflows
- Scientific background
- Customization guide
- Troubleshooting
- Educational use cases
**Best For**: Comprehensive understanding, reference

#### `BAK_SNEPPEN_README.md`
**What**: Technical documentation  
**Lines**: 680  
**Contents**:
- Installation for all platforms
- Detailed usage instructions
- Scientific context and applications
- Performance optimization
- Extension ideas
- Contributing guidelines
**Best For**: Technical details, platform-specific info

#### `MATH_TO_MANIM_PATTERNS.md`
**What**: Design patterns documentation  
**Lines**: 531  
**Contents**:
- 15 design patterns from Math-To-Manim
- Pattern explanations with code
- Combining patterns
- Applying to other models
**Best For**: Learning Manim best practices, developers

#### `PROJECT_SUMMARY.md`
**What**: Executive project summary  
**Lines**: ~550  
**Contents**:
- Deliverables list
- Key features
- Technical specifications
- Goals achieved
- Project metrics
**Best For**: Overview, project managers, quick assessment

#### `INDEX.md`
**What**: This file - navigation guide  
**Contents**: Quick reference to all files and their purpose  
**Best For**: Finding what you need

---

### 📦 Support Files

#### `requirements_bak_sneppen.txt`
**What**: Python dependencies  
**Lines**: 21  
**Install**: `pip install -r requirements_bak_sneppen.txt`  
**Contains**: Core (manim, numpy) + optional (matplotlib, pandas, scipy)

---

## 🎯 Common Tasks → Files to Use

### "I want to render a quick preview"
1. Run: `manim -pql bak_sneppen_3d.py BakSneppenEvolution3D`
2. Or: `.\run_bak_sneppen.ps1 -Quality low`

### "I want to explore interactively"
1. Run: `python bak_sneppen_interactive.py basic`
2. Or for manual control: `python bak_sneppen_interactive.py manual`

### "I want to change parameters"
1. Edit: `bak_sneppen_config.py`
2. Change: `ACTIVE_CONFIG = QuickPreviewConfig` (or other preset)
3. Or customize: Create your own config class

### "I want to analyze simulation data"
1. Render: `manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced`
2. Analyze: `python analyze_bak_sneppen_data.py --latest`
3. Review: Check generated PNG plots and TXT report

### "I want different scene types"
- Main 3D: `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D`
- Histogram: `manim -pqh bak_sneppen_3d.py BakSneppenHistogram`
- Avalanche: `manim -pqh bak_sneppen_3d.py BakSneppenAvalanche`

### "I need help/documentation"
- Quick lookup: `QUICK_REFERENCE.md`
- Full guide: `README_BAK_SNEPPEN_COMPLETE.md`
- Technical: `BAK_SNEPPEN_README.md`
- Patterns: `MATH_TO_MANIM_PATTERNS.md`

### "I want to see all features"
1. Run: `.\demo_all_features.ps1`
2. Follow interactive prompts

### "I want high-quality output"
1. Edit `bak_sneppen_config.py`: `ACTIVE_CONFIG = DetailedConfig`
2. Run: `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D`
3. Or 4K: `manim -pqk bak_sneppen_3d.py BakSneppenEvolution3D`

---

## 📊 File Dependencies

```
Configuration Flow:
bak_sneppen_config.py
    ↓ (imported by)
bak_sneppen_3d_enhanced.py

Data Flow:
bak_sneppen_3d_enhanced.py
    ↓ (produces)
bak_sneppen_data_*.json
    ↓ (analyzed by)
analyze_bak_sneppen_data.py
    ↓ (produces)
*.png plots + *_report.txt

Automation Flow:
run_bak_sneppen.ps1
    ↓ (calls)
manim → bak_sneppen_3d.py or bak_sneppen_3d_enhanced.py

Interactive Flow:
bak_sneppen_interactive.py
    ↓ (standalone, no dependencies on other visualization files)
```

---

## 🎓 Learning Path

### Path 1: Quick Start (5 minutes)
1. Read: `QUICK_REFERENCE.md` (scan)
2. Run: `manim -pql bak_sneppen_3d.py BakSneppenEvolution3D`
3. Watch video

### Path 2: Interactive Exploration (15 minutes)
1. Read: `QUICK_REFERENCE.md` (Interactive section)
2. Run: `python bak_sneppen_interactive.py basic`
3. Run: `python bak_sneppen_interactive.py manual`
4. Experiment with controls

### Path 3: Customization (30 minutes)
1. Read: `README_BAK_SNEPPEN_COMPLETE.md` (Customization section)
2. Edit: `bak_sneppen_config.py` (try different presets)
3. Render: With each preset
4. Compare results

### Path 4: Full Understanding (2 hours)
1. Read: `README_BAK_SNEPPEN_COMPLETE.md` (all)
2. Read: `BAK_SNEPPEN_README.md` (all)
3. Run: `.\demo_all_features.ps1`
4. Experiment: Try different configurations
5. Analyze: Use data export and analysis

### Path 5: Developer/Contributor (4+ hours)
1. Read: All documentation files
2. Read: `MATH_TO_MANIM_PATTERNS.md`
3. Study: Code in all .py files
4. Experiment: Modify and extend
5. Create: Your own variants

---

## 🔍 Finding Specific Information

| I need to know... | Look in... |
|-------------------|------------|
| Command syntax | `QUICK_REFERENCE.md` |
| Installation steps | `BAK_SNEPPEN_README.md` |
| Configuration options | `bak_sneppen_config.py` (comments) |
| Scientific background | `README_BAK_SNEPPEN_COMPLETE.md` (Scientific Context) |
| Troubleshooting | `QUICK_REFERENCE.md` or `README_BAK_SNEPPEN_COMPLETE.md` |
| How patterns work | `MATH_TO_MANIM_PATTERNS.md` |
| Project overview | `PROJECT_SUMMARY.md` |
| Code examples | Any .py file (well-commented) |
| Workflow examples | `README_BAK_SNEPPEN_COMPLETE.md` (Example Workflows) |

---

## 📁 Directory Structure After Rendering

```
StoneyNakoda/
├── Source Files
│   ├── bak_sneppen_3d.py
│   ├── bak_sneppen_3d_enhanced.py
│   ├── bak_sneppen_config.py
│   ├── bak_sneppen_interactive.py
│   └── analyze_bak_sneppen_data.py
│
├── Scripts
│   ├── run_bak_sneppen.ps1
│   └── demo_all_features.ps1
│
├── Documentation
│   ├── INDEX.md (this file)
│   ├── QUICK_REFERENCE.md
│   ├── README_BAK_SNEPPEN_COMPLETE.md
│   ├── BAK_SNEPPEN_README.md
│   ├── MATH_TO_MANIM_PATTERNS.md
│   ├── PROJECT_SUMMARY.md
│   └── requirements_bak_sneppen.txt
│
├── Generated Output (after rendering)
│   ├── media/
│   │   ├── videos/
│   │   │   ├── bak_sneppen_3d/
│   │   │   └── bak_sneppen_3d_enhanced/
│   │   └── images/
│   ├── bak_sneppen_data_*.json
│   ├── bak_sneppen_data_*_fitness_evolution.png
│   ├── bak_sneppen_data_*_fitness_dist.png
│   ├── bak_sneppen_data_*_avalanches.png
│   ├── bak_sneppen_data_*_trajectories.png
│   └── bak_sneppen_data_*_report.txt
│
└── Stoney Nakoda Project Files (original project)
    ├── Dictionaries/
    ├── OpenAIFineTune/
    ├── stoney_rl_grammar/
    └── ... (other files)
```

---

## 🎯 File Selection Decision Tree

```
START: What do you want to do?

├─ Render a video
│  ├─ Standard 3D visualization
│  │  → bak_sneppen_3d.py (BakSneppenEvolution3D)
│  │
│  ├─ With data export for analysis
│  │  → bak_sneppen_3d_enhanced.py (BakSneppenEnhanced)
│  │
│  ├─ Histogram evolution
│  │  → bak_sneppen_3d.py (BakSneppenHistogram)
│  │
│  └─ Close-up avalanche
│     → bak_sneppen_3d.py (BakSneppenAvalanche)
│
├─ Interactive exploration
│  ├─ Automated simulation
│  │  → bak_sneppen_interactive.py basic
│  │
│  ├─ Manual step-by-step
│  │  → bak_sneppen_interactive.py manual
│  │
│  └─ Compare system sizes
│     → bak_sneppen_interactive.py compare
│
├─ Change settings
│  ├─ Use a preset
│  │  → Edit bak_sneppen_config.py (change ACTIVE_CONFIG)
│  │
│  └─ Custom parameters
│     → Edit bak_sneppen_config.py (create custom class)
│
├─ Analyze data
│  └─ From enhanced scene
│     → analyze_bak_sneppen_data.py --latest
│
├─ Learn/Read
│  ├─ Quick reference
│  │  → QUICK_REFERENCE.md
│  │
│  ├─ Full documentation
│  │  → README_BAK_SNEPPEN_COMPLETE.md
│  │
│  ├─ Technical details
│  │  → BAK_SNEPPEN_README.md
│  │
│  ├─ Design patterns
│  │  → MATH_TO_MANIM_PATTERNS.md
│  │
│  └─ Project overview
│     → PROJECT_SUMMARY.md
│
└─ Automate/Demo
   ├─ Easy rendering
   │  → run_bak_sneppen.ps1
   │
   └─ See all features
      → demo_all_features.ps1
```

---

## 📝 Quick Command Reference

```powershell
# Basic rendering
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Enhanced with data
manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced

# Interactive
python bak_sneppen_interactive.py basic

# Analysis
python analyze_bak_sneppen_data.py --latest

# PowerShell launcher
.\run_bak_sneppen.ps1 -Scene basic -Quality high

# Full demo
.\demo_all_features.ps1
```

---

## 🆘 Help Hierarchy

1. **Stuck? Don't know where to start?**
   → Read `QUICK_REFERENCE.md`

2. **Need detailed how-to?**
   → Read `README_BAK_SNEPPEN_COMPLETE.md`

3. **Platform-specific issues?**
   → Read `BAK_SNEPPEN_README.md`

4. **Want to understand the code?**
   → Read `MATH_TO_MANIM_PATTERNS.md`

5. **Still stuck?**
   → Check inline comments in .py files

---

## ✅ Checklist for New Users

- [ ] Install Python 3.8+
- [ ] Install Manim: `pip install manim`
- [ ] Install dependencies: `pip install -r requirements_bak_sneppen.txt`
- [ ] Read `QUICK_REFERENCE.md`
- [ ] Render test: `manim -pql bak_sneppen_3d.py BakSneppenEvolution3D`
- [ ] Try interactive: `python bak_sneppen_interactive.py basic`
- [ ] Read full docs when ready: `README_BAK_SNEPPEN_COMPLETE.md`

---

**Last Updated**: 2025-01-29  
**Total Files**: 12  
**Total Documentation Lines**: ~3,100  
**Total Code Lines**: ~2,400

---

*Navigate with confidence! Each file has a clear purpose.* 🧭


