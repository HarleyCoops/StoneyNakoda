# Bak-Sneppen Model: 3D Manim Visualization

## Overview

This project creates an **epic, rotating 3D visualization** of the Bak-Sneppen evolutionary model, demonstrating self-organized criticality through cascading evolutionary avalanches. The visualization is built using Manim (Mathematical Animation Engine) and inspired by patterns from the Math-To-Manim project.

## The Bak-Sneppen Model

The Bak-Sneppen model is a simple yet profound evolutionary model that demonstrates **self-organized criticality**:

- **Species Arrangement**: N species arranged in a circle
- **Fitness Values**: Each species has a random fitness value between 0 and 1
- **Evolution Rule**: 
  1. Find the species with the lowest fitness
  2. Replace it and its two neighbors with new random fitness values
  3. Repeat
- **Emergent Behavior**: Cascading evolutionary "avalanches" emerge, showing how complex critical behavior arises from simple rules

### Key Insights

- **Self-Organized Criticality**: The system naturally evolves to a critical state without external tuning
- **Power Law Distributions**: Avalanche sizes follow power law distributions
- **Punctuated Equilibrium**: Long periods of stasis interrupted by rapid evolutionary changes
- **Universal Behavior**: Similar patterns emerge across different scales and systems

## Visualization Features

### Main Scene: `BakSneppenEvolution3D`

The primary visualization includes:

1. **3D Rotating View**: Continuously rotating camera for immersive perspective
2. **Species Representation**:
   - Colored spheres (red = low fitness, green = high fitness)
   - Vertical bars showing fitness magnitude
   - Species index labels
3. **Real-time Fitness Graph**: Live bar chart tracking fitness distribution
4. **Animated Avalanches**: Highlighting and smooth transitions during replacements
5. **Iteration Counter**: Tracks simulation progress
6. **Beautiful Finale**: Summary text and final rotation

### Supplementary Scenes

1. **`BakSneppenHistogram`**: Shows fitness distribution evolution over time
2. **`BakSneppenAvalanche`**: Close-up view of a single avalanche cascade event

## Installation

### Prerequisites

1. **Python 3.8+** (tested with Python 3.10)
2. **Manim Community Edition** ([installation guide](https://docs.manim.community/en/stable/installation.html))
3. **LaTeX distribution** (optional, for text rendering)

### Install Manim

#### Windows (PowerShell)

```powershell
# Install via pip
pip install manim

# Or install from source
git clone https://github.com/ManimCommunity/manim.git
cd manim
pip install -e .

# Verify installation
manim --version
```

#### macOS/Linux

```bash
# Install via pip
pip install manim

# Or use conda
conda install -c conda-forge manim

# Verify installation
manim --version
```

### Install Additional Dependencies

```powershell
pip install numpy
```

## Usage

### Render the Main Visualization

```powershell
# High quality (1080p)
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# Medium quality (faster rendering)
manim -pqm bak_sneppen_3d.py BakSneppenEvolution3D

# Low quality (preview)
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D

# 4K quality (slow but stunning)
manim -pqk bak_sneppen_3d.py BakSneppenEvolution3D
```

### Render Supplementary Scenes

```powershell
# Fitness distribution histogram
manim -pqh bak_sneppen_3d.py BakSneppenHistogram

# Close-up avalanche cascade
manim -pqh bak_sneppen_3d.py BakSneppenAvalanche

# Render all scenes
manim -pqh bak_sneppen_3d.py
```

### Command Line Options

- `-p`: Preview/play the animation after rendering
- `-q`: Quality (l=low, m=medium, h=high, k=4K)
- `-s`: Save last frame as image
- `-a`: Render all scenes in file
- `--format=gif`: Export as GIF instead of MP4

## Customization

### Adjust Parameters

Edit the configuration parameters in the `BakSneppenEvolution3D` class:

```python
# In construct() method
self.num_species = 30           # Number of species (default: 30)
self.radius = 4                 # Circle radius (default: 4)
self.sphere_radius = 0.25       # Size of species spheres (default: 0.25)
self.num_iterations = 50        # Simulation iterations (default: 50)
self.animation_speed = 0.5      # Seconds per iteration (default: 0.5)
```

### Modify Camera Behavior

```python
# In setup_camera() method
self.set_camera_orientation(
    phi=75 * DEGREES,          # Vertical angle
    theta=30 * DEGREES         # Horizontal angle
)
self.begin_ambient_camera_rotation(rate=0.15)  # Rotation speed
```

### Change Color Scheme

Modify the `fitness_to_color()` method to use different color gradients:

```python
def fitness_to_color(self, fitness):
    # Example: Blue to Red gradient
    return interpolate_color(BLUE, RED, fitness)
    
    # Example: Purple to Orange gradient
    if fitness < 0.5:
        return interpolate_color(PURPLE, PINK, fitness * 2)
    else:
        return interpolate_color(PINK, ORANGE, (fitness - 0.5) * 2)
```

## Understanding the Code Structure

### Math-To-Manim Inspired Patterns

The code follows best practices from the Math-To-Manim project:

1. **Modular Design**: Separate methods for each visualization component
2. **Clear Documentation**: Comprehensive docstrings explaining each function
3. **Smooth Animations**: Using `LaggedStart`, `Transform`, and interpolation
4. **3D Scene Management**: Proper camera setup and object positioning
5. **Fixed Frame Elements**: UI elements that stay visible during rotation
6. **Color Theory**: Meaningful color mappings for data visualization

### Key Classes and Methods

#### `BakSneppenEvolution3D`

- `construct()`: Main orchestration method
- `setup_camera()`: Configure 3D view
- `create_title()`: Animated title sequence
- `initialize_species()`: Create circular species arrangement
- `fitness_to_color()`: Map fitness values to color gradient
- `create_fitness_graph()`: Real-time fitness distribution chart
- `run_simulation()`: Execute Bak-Sneppen dynamics
- `highlight_species()`: Visual emphasis on affected species
- `replace_species()`: Animated fitness value updates
- `finale()`: Concluding animation sequence

#### `BakSneppenHistogram`

Shows how fitness distribution evolves as the system approaches criticality.

#### `BakSneppenAvalanche`

Provides a dramatic close-up view of the replacement cascade.

## Scientific Context

### Self-Organized Criticality (SOC)

The Bak-Sneppen model is a canonical example of SOC, where:

- **No tuning required**: System naturally evolves to critical state
- **Scale-free behavior**: Similar patterns at different scales
- **Power law distributions**: Avalanche sizes follow P(s) ~ s^(-τ)
- **Long-range correlations**: Events separated in time are correlated

### Real-World Applications

The Bak-Sneppen model has inspired understanding of:

- **Evolution**: Punctuated equilibrium in fossil records
- **Ecology**: Mass extinction events
- **Economics**: Market crashes and financial crises
- **Neuroscience**: Neural avalanches in brain activity
- **Seismology**: Earthquake magnitude distributions (Gutenberg-Richter law)

### Key Publications

1. Bak, P., & Sneppen, K. (1993). "Punctuated equilibrium and criticality in a simple model of evolution." *Physical Review Letters*, 71(24), 4083.
2. Sneppen, K., et al. (1995). "Evolution as a self-organized critical phenomenon." *Proceedings of the National Academy of Sciences*, 92(11), 5209-5213.

## Troubleshooting

### Common Issues

#### 1. LaTeX Not Found

**Error**: `LaTeX Error: Template not found`

**Solution**: Either install LaTeX or use Text objects without LaTeX:

```python
# Replace MathTex with Text
text = Text("Your text here", font_size=24)
```

#### 2. Slow Rendering

**Problem**: High-quality rendering takes too long

**Solution**: Use lower quality for previews:

```powershell
manim -pql bak_sneppen_3d.py BakSneppenEvolution3D
```

#### 3. Memory Issues

**Problem**: System runs out of memory with many species

**Solution**: Reduce `num_species` or `num_iterations`:

```python
self.num_species = 20  # Instead of 30
self.num_iterations = 30  # Instead of 50
```

#### 4. Camera Issues

**Problem**: Objects not visible or camera angle wrong

**Solution**: Adjust camera orientation:

```python
self.set_camera_orientation(phi=60 * DEGREES, theta=0 * DEGREES)
```

## Performance Optimization

### For Faster Rendering

1. **Reduce resolution**: Use `-ql` or `-qm` flags
2. **Decrease iterations**: Lower `num_iterations`
3. **Simplify geometry**: Reduce `resolution` parameter in Sphere/Cylinder creation
4. **Skip graph updates**: Comment out `update_fitness_graph()` calls

### For Better Quality

1. **Increase resolution**: Use `-qk` for 4K
2. **Higher sphere resolution**: Increase `resolution=(20, 20)` to `(30, 30)`
3. **Smoother animations**: Decrease `animation_speed` for slower, smoother transitions
4. **Add motion blur**: Use `--motion_blur` flag (experimental)

## Extension Ideas

### Advanced Visualizations

1. **Multiple Dimensions**: Extend to 2D lattice or 3D volume
2. **Fitness Landscape**: Show fitness as a 3D surface
3. **Avalanche Size Distribution**: Real-time power law histogram
4. **Time Series**: Track fitness evolution for individual species
5. **Critical Point Detection**: Highlight when system reaches criticality

### Interactive Features

1. **Parameter Control**: Use Manim sliders to adjust parameters in real-time
2. **Playback Control**: Add pause/play/rewind functionality
3. **Export Data**: Save fitness values for external analysis
4. **Multiple Models**: Compare Bak-Sneppen with other SOC models

### Educational Enhancements

1. **Step-by-step Explanation**: Add voice-over or text annotations
2. **Mathematical Formulas**: Display relevant equations
3. **Comparison Views**: Split screen showing different parameter sets
4. **Quiz Elements**: Interactive questions about the model

## Integration with Math-To-Manim

This visualization demonstrates key Math-To-Manim principles:

1. **Mathematical Clarity**: Clear visual representation of abstract concepts
2. **Progressive Disclosure**: Building complexity gradually
3. **Color Semantics**: Meaningful use of color to convey information
4. **Animation Pacing**: Balanced timing for comprehension
5. **3D Spatial Reasoning**: Effective use of 3D space for insight

### Using with Math-To-Manim Pipeline

To integrate with the Math-To-Manim automated generation pipeline:

1. **Extract Patterns**: This code demonstrates 3D scientific visualization patterns
2. **Template Generation**: Use as template for similar models (Ising, forest fire, etc.)
3. **Style Guide**: Follow the documentation and structure patterns
4. **Reusable Components**: Extract graph/chart components for other projects

## References

### Manim Resources

- [Manim Community Documentation](https://docs.manim.community/)
- [Manim Community Examples](https://docs.manim.community/en/stable/examples.html)
- [3Blue1Brown (Original Manim)](https://github.com/3b1b/manim)

### Bak-Sneppen Model Resources

- [Wikipedia: Bak-Sneppen Model](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model)
- [Self-Organized Criticality](https://en.wikipedia.org/wiki/Self-organized_criticality)
- [Original Paper (1993)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.71.4083)

### Math-To-Manim

- [Math-To-Manim GitHub Repository](https://github.com/HarleyCoops/Math-To-Manim)

## Contributing

Feel free to extend this visualization! Some ideas:

- Add statistical analysis overlays
- Implement different topology (lattice, random graph)
- Create comparative visualizations with other SOC models
- Add export functionality for simulation data
- Improve performance with optimized rendering

## License

This code is provided as-is for educational and research purposes. When using or modifying, please cite:

- The original Bak-Sneppen model (Bak & Sneppen, 1993)
- Manim Community Edition
- Math-To-Manim project (if applicable)

## Acknowledgments

- **Per Bak & Kim Sneppen**: For the original model
- **Grant Sanderson (3Blue1Brown)**: For creating Manim
- **Manim Community**: For maintaining and improving Manim
- **Math-To-Manim Project**: For inspiration and patterns

---

## Quick Start Summary

```powershell
# 1. Install Manim
pip install manim

# 2. Render the visualization
manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D

# 3. Watch the epic 3D animation!
```

Enjoy exploring self-organized criticality through beautiful mathematical animation! 🌟

