# 🚀 Bak-Sneppen 3D - Quick Reference Card

## One-Line Commands

### Rendering (Pick One)

```powershell
# Basic scene - RECOMMENDED START HERE
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Enhanced with data export
manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced

# PowerShell launcher (Windows)
.\run_bak_sneppen.ps1

# Interactive exploration
python bak_sneppen_interactive.py basic
```

---

## Quality Flags

| Flag | Resolution | Use Case | Render Time |
|------|-----------|----------|-------------|
| `-pql` | 480p | Preview | ~2 min |
| `-pqm` | 720p | Draft | ~5 min |
| `-pqh` | 1080p | Final | ~10 min |
| `-pqk` | 2160p (4K) | Publication | ~30 min |

---

## File Guide

| File | Purpose | Run With |
|------|---------|----------|
| `bak_sneppen_3d.py` | Main 3D scenes | `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D` |
| `bak_sneppen_3d_enhanced.py` | Enhanced + data export | `manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced` |
| `bak_sneppen_config.py` | Configuration presets | Edit `ACTIVE_CONFIG = ...` |
| `bak_sneppen_interactive.py` | Live matplotlib | `python bak_sneppen_interactive.py` |
| `analyze_bak_sneppen_data.py` | Data analysis | `python analyze_bak_sneppen_data.py --latest` |
| `run_bak_sneppen.ps1` | PowerShell launcher | `.\run_bak_sneppen.ps1 -Scene basic` |

---

## Configuration Presets

Edit `ACTIVE_CONFIG` in `bak_sneppen_config.py`:

```python
ACTIVE_CONFIG = BakSneppenConfig       # Default (30 species, 50 iter)
ACTIVE_CONFIG = QuickPreviewConfig     # Fast (15 species, 20 iter)
ACTIVE_CONFIG = DetailedConfig         # High quality (50 species, 100 iter)
ACTIVE_CONFIG = LargeScaleConfig       # Large system (100 species, 200 iter)
ACTIVE_CONFIG = ArtisticConfig         # Purple→Pink→Orange colors
ACTIVE_CONFIG = MinimalistConfig       # Clean minimal style
ACTIVE_CONFIG = EducationalConfig      # Slow pace, more labels
```

---

## Scene Options

| Scene | Description | Command |
|-------|-------------|---------|
| `BakSneppenEvolution3D` | Main 3D rotating visualization | `manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D` |
| `BakSneppenEnhanced` | Enhanced with data tracking | `manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced` |
| `BakSneppenHistogram` | Fitness distribution evolution | `manim -pqh bak_sneppen_3d.py BakSneppenHistogram` |
| `BakSneppenAvalanche` | Close-up cascade visualization | `manim -pqh bak_sneppen_3d.py BakSneppenAvalanche` |

---

## PowerShell Launcher Options

```powershell
.\run_bak_sneppen.ps1 -Scene basic -Quality high
.\run_bak_sneppen.ps1 -Scene enhanced -Quality medium
.\run_bak_sneppen.ps1 -Scene all -Quality low
.\run_bak_sneppen.ps1 -Scene histogram -NoPreview
.\run_bak_sneppen.ps1 -Scene avalanche -ExportGif
```

---

## Interactive Modes

```powershell
python bak_sneppen_interactive.py basic      # 30 species, auto-animate
python bak_sneppen_interactive.py large      # 100 species, fast
python bak_sneppen_interactive.py manual     # Click to advance (SPACE)
python bak_sneppen_interactive.py compare    # Compare system sizes
```

---

## Data Analysis

```powershell
# Analyze most recent data file
python analyze_bak_sneppen_data.py --latest

# Analyze specific file
python analyze_bak_sneppen_data.py bak_sneppen_data_20250129_143022.json

# Generate specific plots only
python analyze_bak_sneppen_data.py --latest --plot fitness
python analyze_bak_sneppen_data.py --latest --plot avalanche
python analyze_bak_sneppen_data.py --latest --plot distribution
```

---

## Common Customizations

### Change Number of Species

Edit `bak_sneppen_config.py`:
```python
NUM_SPECIES = 50  # Change from default 30
```

### Change Color Scheme

```python
LOW_FITNESS_COLOR = BLUE
MID_FITNESS_COLOR = PURPLE
HIGH_FITNESS_COLOR = PINK
```

### Adjust Animation Speed

```python
ITERATION_SPEED = 0.3  # Faster (default 0.5)
ITERATION_SPEED = 1.0  # Slower
```

### Hide/Show Elements

```python
SHOW_SPECIES_LABELS = False       # Hide species numbers
SHOW_FITNESS_GRAPH = False        # Hide live graph
SHOW_ITERATION_COUNTER = False    # Hide counter
```

---

## Export Options

```powershell
# MP4 video (default)
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# GIF animation
manim -pqh --format=gif bak_sneppen_3d.py BakSneppenEvolution3D

# Save last frame as PNG
manim -pqh -s bak_sneppen_3d.py BakSneppenEvolution3D

# Render without playing
manim -qh bak_sneppen_3d.py BakSneppenEvolution3D
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "LaTeX not found" | Install LaTeX or ignore (not critical) |
| Slow rendering | Use `-ql` or reduce `NUM_SPECIES` |
| Out of memory | Lower `SPHERE_RESOLUTION = (10, 10)` |
| Can't see objects | Adjust `CAMERA_PHI` and `CAMERA_THETA` |
| Script won't run (PS) | `Set-ExecutionPolicy RemoteSigned` |

---

## Typical Workflow

1. **Quick test**: `manim -pql bak_sneppen_3d.py BakSneppenEvolution3D`
2. **Interactive explore**: `python bak_sneppen_interactive.py basic`
3. **Configure**: Edit `ACTIVE_CONFIG` in `bak_sneppen_config.py`
4. **Final render**: `manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced`
5. **Analyze data**: `python analyze_bak_sneppen_data.py --latest`

---

## Installation Reminder

```powershell
pip install manim numpy matplotlib
```

---

## Output Locations

- Videos: `media/videos/bak_sneppen_3d/[quality]/`
- Images: `media/images/bak_sneppen_3d/`
- Data: `bak_sneppen_data_YYYYMMDD_HHMMSS.json`
- Plots: `bak_sneppen_data_*_[plot_type].png`

---

## Key Parameters Quick Edit

**In `bak_sneppen_config.py`:**

```python
NUM_SPECIES = 30           # System size
NUM_ITERATIONS = 50        # Simulation length
CIRCLE_RADIUS = 4.0        # Visual scale
ITERATION_SPEED = 0.5      # Animation pace
CAMERA_PHI = 75 * DEGREES  # Vertical angle
AMBIENT_ROTATION_RATE = 0.15  # Rotation speed
```

---

## Help Commands

```powershell
manim --help                    # Manim help
manim --version                 # Check installation
python bak_sneppen_interactive.py --help  # Interactive options
```

---

## 3-Minute Start

```powershell
# 1. Install
pip install manim numpy matplotlib

# 2. Render
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D

# 3. Watch and enjoy! 🎉
```

---

*For full documentation, see `README_BAK_SNEPPEN_COMPLETE.md`*

