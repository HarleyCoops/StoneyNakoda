# 🌀 Bak-Sneppen 3D Visualization Suite

**An epic, rotating 3D visualization of evolutionary self-organized criticality**

![Bak-Sneppen Model](https://img.shields.io/badge/Model-Bak--Sneppen-blue)
![Manim](https://img.shields.io/badge/Manim-Community-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

---

## 🎯 Overview

This comprehensive suite provides **multiple ways** to explore the Bak-Sneppen model—a canonical example of self-organized criticality in evolutionary systems. The model demonstrates how complex critical behavior emerges from simple rules: species arranged in a circle, each with random fitness, where the weakest species and its neighbors are repeatedly replaced with new random values, creating cascading evolutionary avalanches.

### What's Included

1. **🎬 3D Manim Animations** - Cinematic rotating visualizations
2. **⚙️ Configuration System** - Easy parameter tuning with presets
3. **📊 Data Analysis Tools** - Statistical analysis and plotting
4. **🎮 Interactive Simulator** - Real-time exploration with matplotlib
5. **🚀 PowerShell Launcher** - One-command rendering on Windows

---

## 🚀 Quick Start

### Prerequisites

```powershell
# Install Manim Community Edition
pip install manim

# Install additional dependencies
pip install -r requirements_bak_sneppen.txt
```

### Render Your First Animation

```powershell
# Option 1: Direct manim command
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Option 2: Use PowerShell launcher (Windows)
.\run_bak_sneppen.ps1 -Scene basic -Quality high

# Option 3: Try the interactive simulator
python bak_sneppen_interactive.py basic
```

---

## 📂 Project Structure

```
StoneyNakoda/
├── bak_sneppen_3d.py                    # Main 3D visualization scenes
├── bak_sneppen_3d_enhanced.py           # Enhanced version with data export
├── bak_sneppen_config.py                # Centralized configuration system
├── bak_sneppen_interactive.py           # Interactive matplotlib simulator
├── analyze_bak_sneppen_data.py          # Data analysis and plotting
├── run_bak_sneppen.ps1                  # PowerShell launcher script
├── requirements_bak_sneppen.txt         # Python dependencies
├── BAK_SNEPPEN_README.md                # Detailed documentation
└── README_BAK_SNEPPEN_COMPLETE.md       # This file
```

---

## 🎨 Available Scenes

### 1. BakSneppenEvolution3D (Main Scene)

**Epic 3D rotating visualization with all features**

```powershell
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D
```

**Features:**
- ✨ Continuous 3D camera rotation
- 🎨 Color-coded fitness (red=low → green=high)
- 📊 Real-time fitness distribution graph
- 🔢 Iteration counter
- 📍 Species index labels
- 🎯 Animated avalanche highlights
- 🎬 Beautiful finale sequence

**Duration:** ~2-3 minutes  
**Rendering Time:** 5-15 minutes (depending on quality)

---

### 2. BakSneppenEnhanced

**Enhanced version with data tracking and export**

```powershell
manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced
```

**Additional Features:**
- 💾 Automatic JSON data export
- 📈 Statistical summary overlay
- 🎛️ Configuration system integration
- 📊 Extended analytics tracking

**Output:** MP4 video + JSON data file

---

### 3. BakSneppenHistogram

**Fitness distribution evolution over time**

```powershell
manim -pqh bak_sneppen_3d.py BakSneppenHistogram
```

Shows how the fitness distribution changes as the system approaches criticality.

---

### 4. BakSneppenAvalanche

**Close-up view of a single avalanche cascade**

```powershell
manim -pqh bak_sneppen_3d.py BakSneppenAvalanche
```

Dramatic visualization of the replacement cascade with explosion effects.

---

## ⚙️ Configuration System

### Using Presets

Edit `bak_sneppen_config.py` and change the `ACTIVE_CONFIG` line:

```python
# Available presets:
ACTIVE_CONFIG = BakSneppenConfig       # Default balanced settings
ACTIVE_CONFIG = QuickPreviewConfig     # Fast rendering for previews
ACTIVE_CONFIG = DetailedConfig         # High detail for production
ACTIVE_CONFIG = LargeScaleConfig       # 100 species, 200 iterations
ACTIVE_CONFIG = ArtisticConfig         # Beautiful purple→orange colors
ACTIVE_CONFIG = MinimalistConfig       # Clean, minimal aesthetic
ACTIVE_CONFIG = EducationalConfig      # Slower pace, more labels
```

### Key Parameters

```python
# In bak_sneppen_config.py
class BakSneppenConfig:
    NUM_SPECIES = 30           # Number of species
    NUM_ITERATIONS = 50        # Evolutionary steps
    CIRCLE_RADIUS = 4.0        # Arrangement size
    ITERATION_SPEED = 0.5      # Seconds per step
    
    # Colors
    LOW_FITNESS_COLOR = RED
    MID_FITNESS_COLOR = YELLOW
    HIGH_FITNESS_COLOR = GREEN
    
    # Features
    SHOW_SPECIES_LABELS = True
    SHOW_FITNESS_GRAPH = True
    ENABLE_AMBIENT_ROTATION = True
```

---

## 📊 Data Analysis

### Analyze Exported Data

After running the enhanced scene, analyze the exported JSON data:

```powershell
# Analyze specific file
python analyze_bak_sneppen_data.py bak_sneppen_data_20250129_143022.json

# Analyze most recent file
python analyze_bak_sneppen_data.py --latest

# Generate specific plots
python analyze_bak_sneppen_data.py --latest --plot fitness
python analyze_bak_sneppen_data.py --latest --plot avalanche
```

### Generated Outputs

- **Fitness Evolution Plot** - Min/mean fitness over time
- **Fitness Distribution** - Histograms at different time points
- **Avalanche Statistics** - Size distribution and frequency
- **Species Trajectories** - Individual fitness evolution
- **Summary Report** - Text file with key metrics

---

## 🎮 Interactive Exploration

### Live Matplotlib Simulator

```powershell
# Automated demo with 30 species
python bak_sneppen_interactive.py basic

# Large system with 100 species
python bak_sneppen_interactive.py large

# Manual stepping (press SPACE to advance)
python bak_sneppen_interactive.py manual

# Compare different system sizes
python bak_sneppen_interactive.py compare
```

### Features

- **Real-time updates** - Watch evolution unfold
- **Multi-panel display** - Circle, graphs, heatmap, statistics
- **Manual stepping** - Control pace with keyboard
- **Comparison mode** - Side-by-side system sizes

---

## 🎬 Rendering Options

### Quality Settings

```powershell
# Low quality (fast preview)
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D

# Medium quality
manim -pqm bak_sneppen_3d.py BakSneppenEvolution3D

# High quality (1080p) - RECOMMENDED
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# 4K quality (slow but stunning)
manim -pqk bak_sneppen_3d.py BakSneppenEvolution3D
```

### Export Formats

```powershell
# Default: MP4 video
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Export as GIF
manim -pqh --format=gif bak_sneppen_3d.py BakSneppenEvolution3D

# Save last frame as image
manim -pqh -s bak_sneppen_3d.py BakSneppenEvolution3D

# No preview (render only)
manim -qh bak_sneppen_3d.py BakSneppenEvolution3D
```

### PowerShell Launcher

```powershell
# Render with defaults
.\run_bak_sneppen.ps1

# Specify scene and quality
.\run_bak_sneppen.ps1 -Scene enhanced -Quality medium

# Render all scenes
.\run_bak_sneppen.ps1 -Scene all -Quality low

# Export as GIF without preview
.\run_bak_sneppen.ps1 -Scene basic -ExportGif -NoPreview
```

---

## 🔬 Scientific Background

### The Bak-Sneppen Model

Introduced by Per Bak and Kim Sneppen in 1993, this model demonstrates **self-organized criticality** (SOC) in evolutionary systems.

**Core Mechanism:**
1. N species arranged in a circle
2. Each has a random fitness value [0, 1]
3. Find species with minimum fitness
4. Replace it AND its two neighbors with new random values
5. Repeat indefinitely

**Emergent Behavior:**
- System self-organizes to a critical state (~0.667 fitness threshold)
- Power law distribution of avalanche sizes
- Long-range temporal correlations
- No external tuning required

### Applications

The model has inspired understanding of:

- **Evolution** - Punctuated equilibrium in fossil records
- **Ecology** - Mass extinction events
- **Economics** - Market crashes and cascades
- **Neuroscience** - Neural avalanches
- **Seismology** - Earthquake distributions

### Key Publications

1. Bak, P., & Sneppen, K. (1993). "Punctuated equilibrium and criticality in a simple model of evolution." *Physical Review Letters*, 71(24), 4083.

2. Sneppen, K., et al. (1995). "Evolution as a self-organized critical phenomenon." *PNAS*, 92(11), 5209-5213.

---

## 🎓 Educational Use

### For Students

This visualization suite is perfect for:

- **Physics courses** - Statistical mechanics, critical phenomena
- **Biology courses** - Evolutionary dynamics, complex systems
- **Computer Science** - Agent-based modeling, emergence
- **Mathematics** - Stochastic processes, power laws

### Teaching Tips

1. **Start with Interactive** - Use `bak_sneppen_interactive.py` for live exploration
2. **Show the Animation** - Play the full 3D video to capture attention
3. **Analyze Data** - Use exported data to teach statistical analysis
4. **Compare Systems** - Vary parameters to show scaling behavior
5. **Connect to Nature** - Discuss real-world examples (earthquakes, extinctions)

---

## 🛠️ Customization Guide

### Modify Colors

In `bak_sneppen_config.py`:

```python
class MyCustomConfig(BakSneppenConfig):
    LOW_FITNESS_COLOR = PURPLE
    MID_FITNESS_COLOR = PINK
    HIGH_FITNESS_COLOR = ORANGE
```

### Change Camera Behavior

```python
class MyCustomConfig(BakSneppenConfig):
    CAMERA_PHI = 60 * DEGREES        # Lower angle
    CAMERA_THETA = 0 * DEGREES        # Front view
    AMBIENT_ROTATION_RATE = 0.30     # Faster rotation
```

### Adjust System Size

```python
class MyCustomConfig(BakSneppenConfig):
    NUM_SPECIES = 50                 # More species
    NUM_ITERATIONS = 100             # More steps
    CIRCLE_RADIUS = 5.0              # Larger circle
```

### Toggle Features

```python
class MyCustomConfig(BakSneppenConfig):
    SHOW_SPECIES_LABELS = False      # Hide labels
    SHOW_FITNESS_GRAPH = False       # Hide graph
    HIGHLIGHT_AFFECTED_SPECIES = False  # No pulse effect
```

---

## 📈 Performance Tips

### Fast Rendering

```python
# Use QuickPreviewConfig
ACTIVE_CONFIG = QuickPreviewConfig

# Or create custom fast config
class FastConfig(BakSneppenConfig):
    NUM_SPECIES = 15
    NUM_ITERATIONS = 20
    SPHERE_RESOLUTION = (10, 10)
    ITERATION_SPEED = 0.3
```

### High Quality

```python
# Use DetailedConfig
ACTIVE_CONFIG = DetailedConfig

# Or create custom high-quality config
class ProductionConfig(BakSneppenConfig):
    NUM_SPECIES = 50
    NUM_ITERATIONS = 100
    SPHERE_RESOLUTION = (30, 30)
    ITERATION_SPEED = 0.8
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** "LaTeX not found"  
**Solution:** Either install LaTeX or use `Text` instead of `MathTex` in code

**Issue:** Slow rendering  
**Solution:** Use lower quality (`-ql`) or reduce `NUM_SPECIES`/`NUM_ITERATIONS`

**Issue:** Out of memory  
**Solution:** Reduce `SPHERE_RESOLUTION` and `CYLINDER_RESOLUTION`

**Issue:** Objects not visible  
**Solution:** Adjust `CAMERA_PHI` and `CAMERA_THETA` values

**Issue:** PowerShell script won't run  
**Solution:** Enable script execution: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 📚 Resources

### Manim Learning

- [Manim Community Docs](https://docs.manim.community/)
- [Manim Examples Gallery](https://docs.manim.community/en/stable/examples.html)
- [3Blue1Brown Videos](https://www.youtube.com/c/3blue1brown) (Manim creator)

### Bak-Sneppen Model

- [Wikipedia](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model)
- [Self-Organized Criticality](https://en.wikipedia.org/wiki/Self-organized_criticality)
- [Original Paper (1993)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.71.4083)

### Math-To-Manim

- [Math-To-Manim GitHub](https://github.com/HarleyCoops/Math-To-Manim)
- Automated mathematical animation generation

---

## 🎯 Example Workflows

### Workflow 1: Quick Exploration

```powershell
# 1. Run interactive simulator
python bak_sneppen_interactive.py basic

# 2. Render low-quality preview
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D

# 3. If satisfied, render high quality
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D
```

### Workflow 2: Research Analysis

```powershell
# 1. Configure large-scale simulation
# Edit bak_sneppen_config.py: ACTIVE_CONFIG = LargeScaleConfig

# 2. Run enhanced version with data export
manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced

# 3. Analyze exported data
python analyze_bak_sneppen_data.py --latest

# 4. Review plots and statistics
```

### Workflow 3: Educational Presentation

```powershell
# 1. Use educational config (slower, more labels)
# Edit bak_sneppen_config.py: ACTIVE_CONFIG = EducationalConfig

# 2. Render all scenes
.\run_bak_sneppen.ps1 -Scene all -Quality high

# 3. Use in presentation with live interactive demo
python bak_sneppen_interactive.py manual
```

### Workflow 4: Artistic Visualization

```powershell
# 1. Use artistic config (beautiful colors)
# Edit bak_sneppen_config.py: ACTIVE_CONFIG = ArtisticConfig

# 2. Render in 4K
manim -pqk bak_sneppen_3d.py BakSneppenEvolution3D

# 3. Export as GIF for social media
manim -pqh --format=gif bak_sneppen_3d.py BakSneppenEvolution3D
```

---

## 🤝 Contributing

### Extension Ideas

- **Multi-dimensional** - Extend to 2D lattice or 3D volume
- **Fitness Landscape** - Show fitness as 3D surface
- **Power Law Analysis** - Real-time power law fitting
- **Network Topology** - Try different connection patterns
- **Comparative Models** - Side-by-side with other SOC models

### Code Style

- Follow PEP 8 conventions
- Use comprehensive docstrings
- Add type hints where helpful
- Test with multiple configurations
- Document new features in README

---

## 📝 License

This project is provided for **educational and research purposes**. When using or modifying:

- Cite the original Bak-Sneppen model (Bak & Sneppen, 1993)
- Credit Manim Community Edition
- Reference Math-To-Manim project if applicable

---

## 🌟 Acknowledgments

- **Per Bak & Kim Sneppen** - Original model creators
- **Grant Sanderson (3Blue1Brown)** - Manim creator
- **Manim Community** - Maintaining Manim Community Edition
- **Math-To-Manim Project** - Inspiration and patterns

---

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review Manim Community documentation
3. Examine the code comments and docstrings
4. Experiment with different configurations

---

## 🎉 Final Notes

This visualization suite demonstrates how **simple rules create complex behavior**—a fundamental principle in physics, biology, and complex systems science. The Bak-Sneppen model shows that criticality isn't special or fine-tuned; it's a natural emergent property of certain dynamical systems.

**Enjoy exploring self-organized criticality through beautiful mathematical animation!** 🌀✨

---

*Created with ❤️ using Manim Community Edition*  
*Inspired by Math-To-Manim patterns for dynamic scientific visualization*

