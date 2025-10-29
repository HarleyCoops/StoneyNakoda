"""
Bak-Sneppen Model: Enhanced 3D Visualization with Configuration System
=======================================================================

This enhanced version integrates the centralized configuration system
and adds additional features like statistical overlays and data export.

Usage:
    # Use default configuration
    manim -pqh bak_sneppen_3d_enhanced.py BakSneppenEnhanced
    
    # Use quick preview configuration
    # Edit bak_sneppen_config.py and set ACTIVE_CONFIG = QuickPreviewConfig
    manim -pql bak_sneppen_3d_enhanced.py BakSneppenEnhanced
"""

from manim import *
import numpy as np
import random
import json
from datetime import datetime

# Import configuration
try:
    from bak_sneppen_config import get_config
    CONFIG = get_config()
except ImportError:
    print("Warning: Could not import configuration. Using fallback defaults.")
    # Fallback configuration
    class CONFIG:
        NUM_SPECIES = 30
        NUM_ITERATIONS = 50
        CIRCLE_RADIUS = 4.0
        SPHERE_RADIUS = 0.25
        BAR_HEIGHT_SCALE = 2.0
        BAR_RADIUS = 0.05
        CAMERA_PHI = 75 * DEGREES
        CAMERA_THETA = 30 * DEGREES
        AMBIENT_ROTATION_RATE = 0.15
        TITLE_ANIMATION_TIME = 2.0
        INITIAL_WAIT_TIME = 1.0
        SPECIES_CREATION_TIME = 3.0
        ITERATION_SPEED = 0.5
        HIGHLIGHT_DURATION = 0.3
        REPLACEMENT_DURATION = 0.4
        FINALE_ROTATION_TIME = 4.0
        FINALE_WAIT_TIME = 3.0
        LOW_FITNESS_COLOR = RED
        MID_FITNESS_COLOR = YELLOW
        HIGH_FITNESS_COLOR = GREEN
        TITLE_COLOR = BLUE
        SUBTITLE_COLOR = GRAY
        CIRCLE_GUIDE_COLOR = GRAY
        CIRCLE_GUIDE_OPACITY = 0.3
        TITLE_FONT_SIZE = 48
        SUBTITLE_FONT_SIZE = 24
        SPECIES_LABEL_FONT_SIZE = 18
        ITERATION_COUNTER_FONT_SIZE = 24
        GRAPH_TITLE_FONT_SIZE = 20
        GRAPH_LABEL_FONT_SIZE = 18
        GRAPH_X_LENGTH = 5.0
        GRAPH_Y_LENGTH = 2.0
        GRAPH_X_RANGE = [0, 30, 5]
        GRAPH_Y_RANGE = [0, 1, 0.2]
        GRAPH_BAR_OPACITY = 0.7
        GRAPH_UPDATE_TIME = 0.3
        SPHERE_RESOLUTION = (20, 20)
        CYLINDER_RESOLUTION = (8, 2)
        SPHERE_SHEEN = 0.5
        BAR_OPACITY = 0.6
        SHOW_SPECIES_LABELS = True
        SHOW_FITNESS_GRAPH = True
        SHOW_ITERATION_COUNTER = True
        ENABLE_AMBIENT_ROTATION = True
        HIGHLIGHT_AFFECTED_SPECIES = True
        LAG_RATIO_SPECIES = 0.05
        LAG_RATIO_BARS = 0.05
        LAG_RATIO_LABELS = 0.02
        REPLACEMENT_SCALE_FACTOR = 0.3
        
        @staticmethod
        def get_fitness_color(fitness):
            if fitness < 0.5:
                return interpolate_color(RED, YELLOW, fitness * 2)
            else:
                return interpolate_color(YELLOW, GREEN, (fitness - 0.5) * 2)


class BakSneppenEnhanced(ThreeDScene):
    """
    Enhanced Bak-Sneppen visualization with configuration system and data tracking.
    """
    
    def construct(self):
        # Initialize data tracking
        self.simulation_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_species": CONFIG.NUM_SPECIES,
                "num_iterations": CONFIG.NUM_ITERATIONS,
            },
            "iterations": [],
            "statistics": {
                "avalanche_sizes": [],
                "min_fitness_evolution": [],
                "mean_fitness_evolution": [],
            }
        }
        
        # Setup scene
        self.setup_camera()
        self.create_title()
        self.initialize_species()
        
        if CONFIG.SHOW_FITNESS_GRAPH:
            self.create_fitness_graph()
        
        # Run simulation
        self.run_simulation()
        
        # Show statistics
        self.show_statistics()
        
        # Finale
        self.finale()
        
        # Export data
        self.export_simulation_data()
    
    def setup_camera(self):
        """Configure 3D camera from config."""
        self.set_camera_orientation(phi=CONFIG.CAMERA_PHI, theta=CONFIG.CAMERA_THETA)
        if CONFIG.ENABLE_AMBIENT_ROTATION:
            self.begin_ambient_camera_rotation(rate=CONFIG.AMBIENT_ROTATION_RATE)
    
    def create_title(self):
        """Create animated title."""
        title = Text(
            "Bak-Sneppen Model",
            font_size=CONFIG.TITLE_FONT_SIZE,
            color=CONFIG.TITLE_COLOR
        )
        title.to_edge(UP)
        title.fix_in_frame()
        
        subtitle = Text(
            "Self-Organized Criticality in Evolution",
            font_size=CONFIG.SUBTITLE_FONT_SIZE,
            color=CONFIG.SUBTITLE_COLOR
        )
        subtitle.next_to(title, DOWN)
        subtitle.fix_in_frame()
        
        self.play(
            Write(title),
            FadeIn(subtitle, shift=UP),
            run_time=CONFIG.TITLE_ANIMATION_TIME
        )
        self.wait(CONFIG.INITIAL_WAIT_TIME)
        
        self.title_group = VGroup(title, subtitle)
    
    def initialize_species(self):
        """Create circular species arrangement."""
        self.fitness_values = [random.random() for _ in range(CONFIG.NUM_SPECIES)]
        self.species_spheres = VGroup()
        self.fitness_bars = VGroup()
        self.species_labels = VGroup()
        
        for i in range(CONFIG.NUM_SPECIES):
            angle = i * TAU / CONFIG.NUM_SPECIES
            x = CONFIG.CIRCLE_RADIUS * np.cos(angle)
            y = CONFIG.CIRCLE_RADIUS * np.sin(angle)
            z = 0
            
            # Sphere
            sphere = Sphere(
                center=[x, y, z],
                radius=CONFIG.SPHERE_RADIUS,
                resolution=CONFIG.SPHERE_RESOLUTION
            )
            sphere.set_color(CONFIG.get_fitness_color(self.fitness_values[i]))
            sphere.set_sheen(CONFIG.SPHERE_SHEEN, direction=UP)
            
            # Fitness bar
            bar_height = self.fitness_values[i] * CONFIG.BAR_HEIGHT_SCALE
            bar = Cylinder(
                radius=CONFIG.BAR_RADIUS,
                height=bar_height,
                direction=OUT,
                resolution=CONFIG.CYLINDER_RESOLUTION
            )
            bar.move_to([x, y, bar_height / 2])
            bar.set_color(CONFIG.get_fitness_color(self.fitness_values[i]))
            bar.set_opacity(CONFIG.BAR_OPACITY)
            
            self.species_spheres.add(sphere)
            self.fitness_bars.add(bar)
            
            # Labels (optional)
            if CONFIG.SHOW_SPECIES_LABELS:
                label = Integer(i, font_size=CONFIG.SPECIES_LABEL_FONT_SIZE)
                label.move_to([x * 1.15, y * 1.15, 0])
                label.set_color(WHITE)
                self.species_labels.add(label)
        
        # Circle guide
        circle = Circle(
            radius=CONFIG.CIRCLE_RADIUS,
            color=CONFIG.CIRCLE_GUIDE_COLOR,
            stroke_width=2
        )
        circle.set_opacity(CONFIG.CIRCLE_GUIDE_OPACITY)
        
        self.play(Create(circle), run_time=1)
        
        self.play(
            LaggedStart(
                *[GrowFromCenter(s) for s in self.species_spheres],
                lag_ratio=CONFIG.LAG_RATIO_SPECIES
            ),
            LaggedStart(
                *[GrowFromPoint(b, b.get_bottom()) for b in self.fitness_bars],
                lag_ratio=CONFIG.LAG_RATIO_BARS
            ),
            run_time=CONFIG.SPECIES_CREATION_TIME
        )
        
        if CONFIG.SHOW_SPECIES_LABELS:
            self.play(
                LaggedStart(
                    *[FadeIn(l) for l in self.species_labels],
                    lag_ratio=CONFIG.LAG_RATIO_LABELS
                ),
                run_time=1
            )
        
        self.circle_guide = circle
        self.wait(1)
    
    def create_fitness_graph(self):
        """Create fitness distribution graph."""
        graph_axes = Axes(
            x_range=CONFIG.GRAPH_X_RANGE,
            y_range=CONFIG.GRAPH_Y_RANGE,
            x_length=CONFIG.GRAPH_X_LENGTH,
            y_length=CONFIG.GRAPH_Y_LENGTH,
            axis_config={"include_numbers": False, "stroke_width": 2},
        )
        graph_axes.to_corner(DR, buff=0.5)
        graph_axes.fix_in_frame()
        
        x_label = Text("Species", font_size=CONFIG.GRAPH_LABEL_FONT_SIZE)
        x_label.next_to(graph_axes.x_axis, DOWN, buff=0.2)
        x_label.fix_in_frame()
        
        y_label = Text("Fitness", font_size=CONFIG.GRAPH_LABEL_FONT_SIZE)
        y_label.next_to(graph_axes.y_axis, LEFT, buff=0.2)
        y_label.fix_in_frame()
        
        graph_title = Text(
            "Fitness Distribution",
            font_size=CONFIG.GRAPH_TITLE_FONT_SIZE,
            color=CONFIG.TITLE_COLOR
        )
        graph_title.next_to(graph_axes, UP, buff=0.2)
        graph_title.fix_in_frame()
        
        self.play(
            Create(graph_axes),
            Write(graph_title),
            FadeIn(x_label),
            FadeIn(y_label),
            run_time=1
        )
        
        self.graph_axes = graph_axes
        self.graph_title = graph_title
        self.graph_labels = VGroup(x_label, y_label)
        self.update_fitness_graph()
    
    def update_fitness_graph(self):
        """Update fitness distribution graph."""
        bars = VGroup()
        bar_width = self.graph_axes.x_axis.get_unit_size() * 0.8
        
        for i, fitness in enumerate(self.fitness_values):
            x_pos = self.graph_axes.c2p(i, 0)[0]
            y_height = fitness * self.graph_axes.y_axis.get_unit_size()
            
            bar = Rectangle(
                width=bar_width,
                height=max(y_height, 0.01),  # Prevent zero height
                fill_color=CONFIG.get_fitness_color(fitness),
                fill_opacity=CONFIG.GRAPH_BAR_OPACITY,
                stroke_width=1,
                stroke_color=WHITE
            )
            bar.move_to(self.graph_axes.c2p(i, 0), aligned_edge=DOWN)
            bar.fix_in_frame()
            bars.add(bar)
        
        if hasattr(self, 'fitness_chart'):
            self.play(
                Transform(self.fitness_chart, bars),
                run_time=CONFIG.GRAPH_UPDATE_TIME
            )
        else:
            self.fitness_chart = bars
            self.play(FadeIn(self.fitness_chart), run_time=0.5)
    
    def run_simulation(self):
        """Execute Bak-Sneppen dynamics with data tracking."""
        if CONFIG.SHOW_ITERATION_COUNTER:
            iteration_counter = Integer(0, font_size=CONFIG.ITERATION_COUNTER_FONT_SIZE)
            iteration_counter.to_corner(UL, buff=1)
            iteration_counter.fix_in_frame()
            
            iteration_label = Text("Iteration: ", font_size=CONFIG.ITERATION_COUNTER_FONT_SIZE)
            iteration_label.next_to(iteration_counter, LEFT, buff=0.2)
            iteration_label.fix_in_frame()
            
            self.play(FadeIn(iteration_label), FadeIn(iteration_counter), run_time=0.5)
        
        self.wait(2)
        
        for iteration in range(CONFIG.NUM_ITERATIONS):
            # Find weakest
            min_fitness = min(self.fitness_values)
            min_fitness_idx = self.fitness_values.index(min_fitness)
            
            # Neighbors
            left_neighbor = (min_fitness_idx - 1) % CONFIG.NUM_SPECIES
            right_neighbor = (min_fitness_idx + 1) % CONFIG.NUM_SPECIES
            affected_indices = [left_neighbor, min_fitness_idx, right_neighbor]
            
            # Track statistics
            self.simulation_data["statistics"]["min_fitness_evolution"].append(min_fitness)
            self.simulation_data["statistics"]["mean_fitness_evolution"].append(
                np.mean(self.fitness_values)
            )
            self.simulation_data["statistics"]["avalanche_sizes"].append(len(affected_indices))
            
            # Record iteration data
            self.simulation_data["iterations"].append({
                "iteration": iteration,
                "min_fitness": min_fitness,
                "min_fitness_idx": min_fitness_idx,
                "affected_indices": affected_indices,
                "fitness_snapshot": self.fitness_values.copy()
            })
            
            # Highlight
            if CONFIG.HIGHLIGHT_AFFECTED_SPECIES:
                self.highlight_species(affected_indices)
            
            # Replace
            new_fitness_values = [random.random() for _ in affected_indices]
            self.replace_species(affected_indices, new_fitness_values)
            
            # Update UI
            if CONFIG.SHOW_ITERATION_COUNTER:
                iteration_counter.set_value(iteration + 1)
            
            if CONFIG.SHOW_FITNESS_GRAPH:
                self.update_fitness_graph()
            
            self.wait(CONFIG.ITERATION_SPEED)
    
    def highlight_species(self, indices):
        """Highlight species to be replaced."""
        highlight_spheres = VGroup()
        
        for idx in indices:
            sphere = self.species_spheres[idx]
            pulse = sphere.copy()
            pulse.set_color(WHITE)
            pulse.set_opacity(0.8)
            highlight_spheres.add(pulse)
        
        self.play(
            *[s.animate.scale(1.5).set_opacity(0.3) for s in highlight_spheres],
            run_time=CONFIG.HIGHLIGHT_DURATION
        )
        self.play(
            *[FadeOut(s) for s in highlight_spheres],
            run_time=0.2
        )
    
    def replace_species(self, indices, new_fitness_values):
        """Replace species with new fitness."""
        animations = []
        
        for i, idx in enumerate(indices):
            new_fitness = new_fitness_values[i]
            self.fitness_values[idx] = new_fitness
            
            new_color = CONFIG.get_fitness_color(new_fitness)
            sphere = self.species_spheres[idx]
            animations.append(sphere.animate.set_color(new_color))
            
            bar = self.fitness_bars[idx]
            angle = idx * TAU / CONFIG.NUM_SPECIES
            x = CONFIG.CIRCLE_RADIUS * np.cos(angle)
            y = CONFIG.CIRCLE_RADIUS * np.sin(angle)
            
            new_bar_height = new_fitness * CONFIG.BAR_HEIGHT_SCALE
            new_bar = Cylinder(
                radius=CONFIG.BAR_RADIUS,
                height=new_bar_height,
                direction=OUT,
                resolution=CONFIG.CYLINDER_RESOLUTION
            )
            new_bar.move_to([x, y, new_bar_height / 2])
            new_bar.set_color(new_color)
            new_bar.set_opacity(CONFIG.BAR_OPACITY)
            
            animations.append(Transform(bar, new_bar))
        
        self.play(*animations, run_time=CONFIG.REPLACEMENT_DURATION)
    
    def show_statistics(self):
        """Display summary statistics."""
        stats_data = self.simulation_data["statistics"]
        
        # Calculate statistics
        min_fitness_final = stats_data["min_fitness_evolution"][-1]
        mean_fitness_final = stats_data["mean_fitness_evolution"][-1]
        total_replacements = sum(stats_data["avalanche_sizes"])
        
        stats_text = VGroup(
            Text(
                f"Final Min Fitness: {min_fitness_final:.3f}",
                font_size=20,
                color=CONFIG.LOW_FITNESS_COLOR
            ),
            Text(
                f"Final Mean Fitness: {mean_fitness_final:.3f}",
                font_size=20,
                color=CONFIG.MID_FITNESS_COLOR
            ),
            Text(
                f"Total Replacements: {total_replacements}",
                font_size=20,
                color=CONFIG.HIGH_FITNESS_COLOR
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        stats_text.to_corner(UL, buff=0.5)
        stats_text.fix_in_frame()
        
        self.play(FadeIn(stats_text, shift=RIGHT), run_time=1)
        self.wait(2)
        self.play(FadeOut(stats_text), run_time=0.5)
    
    def finale(self):
        """Final animation sequence."""
        if CONFIG.ENABLE_AMBIENT_ROTATION:
            self.stop_ambient_camera_rotation()
        
        summary = VGroup(
            Text("Cascading Avalanches Observed", font_size=36, color=GREEN),
            Text("Self-Organized Criticality Emerges", font_size=28, color=YELLOW),
            Text("From Simple Evolutionary Rules", font_size=24, color=GRAY)
        ).arrange(DOWN, buff=0.3)
        summary.fix_in_frame()
        
        if CONFIG.SHOW_FITNESS_GRAPH:
            self.play(
                FadeOut(self.graph_axes),
                FadeOut(self.graph_title),
                FadeOut(self.graph_labels),
                FadeOut(self.fitness_chart),
                run_time=1
            )
        
        self.play(
            Rotate(
                VGroup(self.species_spheres, self.fitness_bars, self.circle_guide),
                angle=TAU,
                axis=OUT,
                run_time=CONFIG.FINALE_ROTATION_TIME,
                rate_func=smooth
            )
        )
        
        self.play(
            FadeOut(self.title_group),
            FadeIn(summary, shift=UP),
            run_time=2
        )
        
        self.wait(CONFIG.FINALE_WAIT_TIME)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=2)
    
    def export_simulation_data(self):
        """Export simulation data to JSON file."""
        try:
            filename = f"bak_sneppen_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.simulation_data, f, indent=2)
            print(f"\nSimulation data exported to: {filename}")
        except Exception as e:
            print(f"\nWarning: Could not export simulation data: {e}")


# Additional scene configurations can be created by changing ACTIVE_CONFIG
# in bak_sneppen_config.py before running manim

