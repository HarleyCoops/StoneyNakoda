"""
Configuration file for Bak-Sneppen 3D Visualization
====================================================

Centralized configuration for easy parameter tuning and experimentation.
Modify these values to customize the visualization without editing the main code.
"""

from manim import *


class BakSneppenConfig:
    """Configuration parameters for the Bak-Sneppen visualization."""
    
    # Simulation Parameters
    # ---------------------
    NUM_SPECIES = 30                    # Number of species in the circle
    NUM_ITERATIONS = 50                 # Number of evolutionary steps to simulate
    INITIAL_FITNESS_RANGE = (0.0, 1.0)  # Range for random fitness values
    
    # Visual Parameters
    # -----------------
    CIRCLE_RADIUS = 4.0                 # Radius of species arrangement
    SPHERE_RADIUS = 0.25                # Size of species spheres
    BAR_HEIGHT_SCALE = 2.0              # Scale factor for fitness bars
    BAR_RADIUS = 0.05                   # Radius of fitness bar cylinders
    
    # Camera Parameters
    # -----------------
    CAMERA_PHI = 75 * DEGREES           # Vertical camera angle
    CAMERA_THETA = 30 * DEGREES         # Horizontal camera angle
    AMBIENT_ROTATION_RATE = 0.15        # Speed of automatic rotation
    
    # Animation Timing
    # ----------------
    TITLE_ANIMATION_TIME = 2.0          # Duration of title animation
    INITIAL_WAIT_TIME = 1.0             # Pause after title
    SPECIES_CREATION_TIME = 3.0         # Time to create all species
    ITERATION_SPEED = 0.5               # Seconds per evolutionary iteration
    HIGHLIGHT_DURATION = 0.3            # Time to highlight weakest species
    REPLACEMENT_DURATION = 0.4          # Time to update fitness values
    FINALE_ROTATION_TIME = 4.0          # Final rotation duration
    FINALE_WAIT_TIME = 3.0              # Pause before fadeout
    
    # Color Scheme
    # ------------
    LOW_FITNESS_COLOR = RED             # Color for low fitness (0.0)
    MID_FITNESS_COLOR = YELLOW          # Color for medium fitness (0.5)
    HIGH_FITNESS_COLOR = GREEN          # Color for high fitness (1.0)
    BACKGROUND_COLOR = BLACK            # Scene background
    TITLE_COLOR = BLUE                  # Title text color
    SUBTITLE_COLOR = GRAY               # Subtitle text color
    CIRCLE_GUIDE_COLOR = GRAY           # Circle outline color
    CIRCLE_GUIDE_OPACITY = 0.3          # Circle outline transparency
    
    # Text Parameters
    # ---------------
    TITLE_FONT_SIZE = 48
    SUBTITLE_FONT_SIZE = 24
    SPECIES_LABEL_FONT_SIZE = 18
    ITERATION_COUNTER_FONT_SIZE = 24
    GRAPH_TITLE_FONT_SIZE = 20
    GRAPH_LABEL_FONT_SIZE = 18
    SUMMARY_FONT_SIZE = 36
    
    # Graph Parameters
    # ----------------
    GRAPH_X_LENGTH = 5.0                # Width of fitness graph
    GRAPH_Y_LENGTH = 2.0                # Height of fitness graph
    GRAPH_X_RANGE = [0, NUM_SPECIES, 5] # X-axis range and tick interval
    GRAPH_Y_RANGE = [0, 1, 0.2]         # Y-axis range and tick interval
    GRAPH_BAR_OPACITY = 0.7             # Transparency of histogram bars
    GRAPH_UPDATE_TIME = 0.3             # Duration of graph updates
    
    # 3D Object Parameters
    # --------------------
    SPHERE_RESOLUTION = (20, 20)        # U,V resolution for spheres
    CYLINDER_RESOLUTION = (8, 2)        # Circular segments, height segments
    SPHERE_SHEEN = 0.5                  # Shininess of spheres
    BAR_OPACITY = 0.6                   # Transparency of fitness bars
    
    # Behavior Flags
    # --------------
    SHOW_SPECIES_LABELS = True          # Display species index numbers
    SHOW_FITNESS_GRAPH = True           # Display real-time fitness graph
    SHOW_ITERATION_COUNTER = True       # Display iteration counter
    ENABLE_AMBIENT_ROTATION = True      # Enable continuous camera rotation
    HIGHLIGHT_AFFECTED_SPECIES = True   # Pulse effect on replaced species
    
    # Advanced Parameters
    # -------------------
    LAG_RATIO_SPECIES = 0.05            # Stagger effect for species creation
    LAG_RATIO_BARS = 0.05               # Stagger effect for bar creation
    LAG_RATIO_LABELS = 0.02             # Stagger effect for label creation
    EXPLOSION_SCALE = 3.0               # Scale factor for highlight effect
    REPLACEMENT_SCALE_FACTOR = 0.3      # How much species shrink during replacement
    
    # Statistical Parameters
    # ----------------------
    HISTOGRAM_BINS = 10                 # Number of bins for fitness histogram
    MOVING_AVERAGE_WINDOW = 5           # Window size for smoothing (if implemented)
    
    @classmethod
    def get_fitness_color(cls, fitness: float) -> ManimColor:
        """
        Convert fitness value to color using configured gradient.
        
        Args:
            fitness: Fitness value between 0 and 1
            
        Returns:
            Interpolated color based on fitness value
        """
        if fitness < 0.5:
            # Low to medium fitness
            t = fitness * 2
            return interpolate_color(cls.LOW_FITNESS_COLOR, cls.MID_FITNESS_COLOR, t)
        else:
            # Medium to high fitness
            t = (fitness - 0.5) * 2
            return interpolate_color(cls.MID_FITNESS_COLOR, cls.HIGH_FITNESS_COLOR, t)
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        if cls.NUM_SPECIES < 3:
            errors.append("NUM_SPECIES must be at least 3")
        
        if cls.NUM_ITERATIONS < 1:
            errors.append("NUM_ITERATIONS must be at least 1")
        
        if cls.CIRCLE_RADIUS <= 0:
            errors.append("CIRCLE_RADIUS must be positive")
        
        if cls.SPHERE_RADIUS <= 0:
            errors.append("SPHERE_RADIUS must be positive")
        
        if not (0.0 <= cls.INITIAL_FITNESS_RANGE[0] < cls.INITIAL_FITNESS_RANGE[1] <= 1.0):
            errors.append("INITIAL_FITNESS_RANGE must be within [0, 1] with min < max")
        
        if errors:
            print("Configuration Validation Errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Preset Configurations
# =====================

class QuickPreviewConfig(BakSneppenConfig):
    """Fast rendering for quick previews."""
    NUM_SPECIES = 15
    NUM_ITERATIONS = 20
    SPHERE_RESOLUTION = (10, 10)
    CYLINDER_RESOLUTION = (6, 2)
    SPECIES_CREATION_TIME = 1.5
    ITERATION_SPEED = 0.3
    SHOW_SPECIES_LABELS = False


class DetailedConfig(BakSneppenConfig):
    """High detail for final production."""
    NUM_SPECIES = 50
    NUM_ITERATIONS = 100
    SPHERE_RESOLUTION = (30, 30)
    CYLINDER_RESOLUTION = (16, 4)
    SPECIES_CREATION_TIME = 5.0
    ITERATION_SPEED = 0.6


class LargeScaleConfig(BakSneppenConfig):
    """Large system for statistical analysis."""
    NUM_SPECIES = 100
    NUM_ITERATIONS = 200
    CIRCLE_RADIUS = 6.0
    SPHERE_RADIUS = 0.15
    BAR_RADIUS = 0.03
    SPECIES_LABEL_FONT_SIZE = 12
    SHOW_SPECIES_LABELS = False
    ITERATION_SPEED = 0.2


class ArtisticConfig(BakSneppenConfig):
    """Beautiful color scheme for artistic visualization."""
    LOW_FITNESS_COLOR = PURPLE
    MID_FITNESS_COLOR = PINK
    HIGH_FITNESS_COLOR = ORANGE
    BACKGROUND_COLOR = "#0a0a0a"
    TITLE_COLOR = GOLD
    SUBTITLE_COLOR = TEAL
    SPHERE_SHEEN = 0.8
    AMBIENT_ROTATION_RATE = 0.08
    ITERATION_SPEED = 0.7


class MinimalistConfig(BakSneppenConfig):
    """Clean, minimal visualization."""
    SHOW_SPECIES_LABELS = False
    SHOW_ITERATION_COUNTER = False
    LOW_FITNESS_COLOR = GRAY_A
    MID_FITNESS_COLOR = GRAY
    HIGH_FITNESS_COLOR = WHITE
    CIRCLE_GUIDE_OPACITY = 0.1
    BAR_OPACITY = 0.4
    ITERATION_SPEED = 0.4


class EducationalConfig(BakSneppenConfig):
    """Slower pace for educational contexts."""
    NUM_SPECIES = 20
    NUM_ITERATIONS = 30
    INITIAL_WAIT_TIME = 2.0
    ITERATION_SPEED = 1.0
    HIGHLIGHT_DURATION = 0.5
    REPLACEMENT_DURATION = 0.6
    TITLE_ANIMATION_TIME = 3.0
    SHOW_SPECIES_LABELS = True
    SHOW_ITERATION_COUNTER = True


# Active Configuration Selection
# ===============================

# Change this line to switch between configurations
ACTIVE_CONFIG = BakSneppenConfig  # Options: BakSneppenConfig, QuickPreviewConfig, 
                                  #          DetailedConfig, LargeScaleConfig,
                                  #          ArtisticConfig, MinimalistConfig,
                                  #          EducationalConfig

# Validate configuration on import
if __name__ != "__main__":
    if not ACTIVE_CONFIG.validate_config():
        print("\nWarning: Configuration validation failed!")
        print("Using default BakSneppenConfig instead.")
        ACTIVE_CONFIG = BakSneppenConfig


# Convenience function for external use
# ======================================

def get_config():
    """
    Get the active configuration.
    
    Returns:
        Active configuration class
    """
    return ACTIVE_CONFIG


def list_available_configs():
    """
    List all available configuration presets.
    
    Returns:
        Dictionary mapping config names to config classes
    """
    return {
        "default": BakSneppenConfig,
        "quick": QuickPreviewConfig,
        "detailed": DetailedConfig,
        "large": LargeScaleConfig,
        "artistic": ArtisticConfig,
        "minimalist": MinimalistConfig,
        "educational": EducationalConfig,
    }


def switch_config(config_name: str):
    """
    Switch to a different configuration preset.
    
    Args:
        config_name: Name of the configuration to use
        
    Returns:
        The selected configuration class
        
    Raises:
        ValueError: If config_name is not recognized
    """
    global ACTIVE_CONFIG
    
    configs = list_available_configs()
    
    if config_name.lower() not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown configuration '{config_name}'. "
            f"Available options: {available}"
        )
    
    ACTIVE_CONFIG = configs[config_name.lower()]
    
    if not ACTIVE_CONFIG.validate_config():
        print(f"\nWarning: Configuration '{config_name}' validation failed!")
        print("Reverting to default configuration.")
        ACTIVE_CONFIG = BakSneppenConfig
    
    return ACTIVE_CONFIG


# Example usage in main code:
# ---------------------------
# from bak_sneppen_config import get_config
# 
# config = get_config()
# self.num_species = config.NUM_SPECIES
# self.radius = config.CIRCLE_RADIUS
# ...

