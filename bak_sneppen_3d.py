"""
Bak-Sneppen Model: 3D Rotating Visualization
=============================================

An epic 3D visualization of the Bak-Sneppen evolutionary model demonstrating
self-organized criticality. Species are arranged in a circle with fitness values
represented by colored spheres. The weakest species and neighbors undergo
cascading evolutionary avalanches.

Inspired by Math-To-Manim patterns for dynamic mathematical visualization.
"""

from manim import *
import numpy as np
import random


class BakSneppenEvolution3D(ThreeDScene):
    """
    Main scene for the Bak-Sneppen model visualization.
    
    Features:
    - 3D rotating view
    - Species arranged in a circle
    - Color-coded fitness values (red=low, green=high)
    - Animated avalanche cascades
    - Real-time fitness distribution tracking
    """
    
    def construct(self):
        # Configuration parameters
        self.num_species = 30
        self.radius = 4
        self.sphere_radius = 0.25
        self.num_iterations = 50
        self.animation_speed = 0.5
        
        # Initialize the scene
        self.setup_camera()
        self.create_title()
        self.initialize_species()
        self.create_fitness_graph()
        
        # Run the evolutionary simulation
        self.run_simulation()
        
        # Final rotation and conclusion
        self.finale()
    
    def setup_camera(self):
        """Configure the 3D camera for optimal viewing."""
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15)
    
    def create_title(self):
        """Create animated title and description."""
        title = Text("Bak-Sneppen Model", font_size=48, color=BLUE)
        title.to_edge(UP)
        title.fix_in_frame()
        
        subtitle = Text(
            "Self-Organized Criticality in Evolution",
            font_size=24,
            color=GRAY
        )
        subtitle.next_to(title, DOWN)
        subtitle.fix_in_frame()
        
        self.play(
            Write(title),
            FadeIn(subtitle, shift=UP),
            run_time=2
        )
        self.wait(1)
        
        # Store for later reference
        self.title_group = VGroup(title, subtitle)
    
    def initialize_species(self):
        """Create the circular arrangement of species with random fitness."""
        self.fitness_values = [random.random() for _ in range(self.num_species)]
        self.species_spheres = VGroup()
        self.fitness_bars = VGroup()
        self.species_labels = VGroup()
        
        for i in range(self.num_species):
            angle = i * TAU / self.num_species
            
            # Position on circle
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = 0
            
            # Create sphere for species
            sphere = Sphere(
                center=[x, y, z],
                radius=self.sphere_radius,
                resolution=(20, 20)
            )
            sphere.set_color(self.fitness_to_color(self.fitness_values[i]))
            sphere.set_sheen(0.5, direction=UP)
            
            # Create vertical fitness bar
            bar_height = self.fitness_values[i] * 2
            bar = Cylinder(
                radius=0.05,
                height=bar_height,
                direction=OUT,
                resolution=(8, 2)
            )
            bar.move_to([x, y, bar_height / 2])
            bar.set_color(self.fitness_to_color(self.fitness_values[i]))
            bar.set_opacity(0.6)
            
            # Species index label
            label = Integer(i, font_size=18)
            label.move_to([x * 1.15, y * 1.15, 0])
            label.set_color(WHITE)
            
            self.species_spheres.add(sphere)
            self.fitness_bars.add(bar)
            self.species_labels.add(label)
        
        # Add connecting circle
        circle = Circle(radius=self.radius, color=GRAY, stroke_width=2)
        circle.set_opacity(0.3)
        
        # Animate creation
        self.play(
            Create(circle),
            run_time=1
        )
        
        self.play(
            LaggedStart(
                *[GrowFromCenter(sphere) for sphere in self.species_spheres],
                lag_ratio=0.05
            ),
            LaggedStart(
                *[GrowFromPoint(bar, bar.get_bottom()) for bar in self.fitness_bars],
                lag_ratio=0.05
            ),
            run_time=3
        )
        
        self.play(
            LaggedStart(
                *[FadeIn(label) for label in self.species_labels],
                lag_ratio=0.02
            ),
            run_time=1
        )
        
        self.circle_guide = circle
        self.wait(1)
    
    def fitness_to_color(self, fitness):
        """
        Convert fitness value to color gradient.
        Red (low fitness) -> Yellow (medium) -> Green (high fitness)
        """
        if fitness < 0.5:
            # Red to Yellow
            t = fitness * 2
            return interpolate_color(RED, YELLOW, t)
        else:
            # Yellow to Green
            t = (fitness - 0.5) * 2
            return interpolate_color(YELLOW, GREEN, t)
    
    def create_fitness_graph(self):
        """Create a live fitness distribution graph."""
        # Graph configuration
        graph_axes = Axes(
            x_range=[0, self.num_species, 5],
            y_range=[0, 1, 0.2],
            x_length=5,
            y_length=2,
            axis_config={"include_numbers": False, "stroke_width": 2},
        )
        graph_axes.to_corner(DR, buff=0.5)
        graph_axes.fix_in_frame()
        
        # Labels
        x_label = Text("Species", font_size=18).next_to(graph_axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Fitness", font_size=18).next_to(graph_axes.y_axis, LEFT, buff=0.2)
        x_label.fix_in_frame()
        y_label.fix_in_frame()
        
        graph_title = Text("Fitness Distribution", font_size=20, color=BLUE)
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
        """Update the fitness distribution graph."""
        # Create bar chart
        bars = VGroup()
        bar_width = self.graph_axes.x_axis.get_unit_size() * 0.8
        
        for i, fitness in enumerate(self.fitness_values):
            x_pos = self.graph_axes.c2p(i, 0)[0]
            y_height = fitness * self.graph_axes.y_axis.get_unit_size()
            
            bar = Rectangle(
                width=bar_width,
                height=y_height,
                fill_color=self.fitness_to_color(fitness),
                fill_opacity=0.7,
                stroke_width=1,
                stroke_color=WHITE
            )
            bar.move_to(self.graph_axes.c2p(i, 0), aligned_edge=DOWN)
            bar.fix_in_frame()
            bars.add(bar)
        
        if hasattr(self, 'fitness_chart'):
            self.play(
                Transform(self.fitness_chart, bars),
                run_time=0.3
            )
        else:
            self.fitness_chart = bars
            self.play(FadeIn(self.fitness_chart), run_time=0.5)
    
    def run_simulation(self):
        """Execute the Bak-Sneppen evolutionary dynamics."""
        iteration_counter = Integer(0, font_size=24)
        iteration_counter.to_corner(UL, buff=1)
        iteration_counter.fix_in_frame()
        
        iteration_label = Text("Iteration: ", font_size=24)
        iteration_label.next_to(iteration_counter, LEFT, buff=0.2)
        iteration_label.fix_in_frame()
        
        self.play(
            FadeIn(iteration_label),
            FadeIn(iteration_counter),
            run_time=0.5
        )
        
        # Pause to show initial state
        self.wait(2)
        
        for iteration in range(self.num_iterations):
            # Find weakest species
            min_fitness_idx = self.fitness_values.index(min(self.fitness_values))
            
            # Determine neighbors (circular)
            left_neighbor = (min_fitness_idx - 1) % self.num_species
            right_neighbor = (min_fitness_idx + 1) % self.num_species
            
            affected_indices = [left_neighbor, min_fitness_idx, right_neighbor]
            
            # Highlight the weakest and neighbors
            self.highlight_species(affected_indices)
            
            # Update fitness values
            new_fitness_values = [random.random() for _ in affected_indices]
            
            # Animate the replacement
            self.replace_species(affected_indices, new_fitness_values)
            
            # Update counter
            iteration_counter.set_value(iteration + 1)
            
            # Update graph
            self.update_fitness_graph()
            
            # Short pause between iterations
            self.wait(self.animation_speed)
    
    def highlight_species(self, indices):
        """Highlight species that will be replaced."""
        highlight_spheres = VGroup()
        
        for idx in indices:
            sphere = self.species_spheres[idx]
            pulse = sphere.copy()
            pulse.set_color(WHITE)
            pulse.set_opacity(0.8)
            highlight_spheres.add(pulse)
        
        self.play(
            *[sphere.animate.scale(1.5).set_opacity(0.3) for sphere in highlight_spheres],
            run_time=0.3
        )
        self.play(
            *[FadeOut(sphere) for sphere in highlight_spheres],
            run_time=0.2
        )
    
    def replace_species(self, indices, new_fitness_values):
        """Replace species with new fitness values."""
        animations = []
        
        for i, idx in enumerate(indices):
            new_fitness = new_fitness_values[i]
            self.fitness_values[idx] = new_fitness
            
            # Update sphere color
            new_color = self.fitness_to_color(new_fitness)
            sphere = self.species_spheres[idx]
            animations.append(sphere.animate.set_color(new_color))
            
            # Update fitness bar
            bar = self.fitness_bars[idx]
            angle = idx * TAU / self.num_species
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            
            new_bar_height = new_fitness * 2
            new_bar = Cylinder(
                radius=0.05,
                height=new_bar_height,
                direction=OUT,
                resolution=(8, 2)
            )
            new_bar.move_to([x, y, new_bar_height / 2])
            new_bar.set_color(new_color)
            new_bar.set_opacity(0.6)
            
            animations.append(Transform(bar, new_bar))
        
        self.play(*animations, run_time=0.4)
    
    def finale(self):
        """Final animation sequence."""
        # Speed up rotation
        self.stop_ambient_camera_rotation()
        
        # Create summary text
        summary = VGroup(
            Text("Cascading Avalanches Observed", font_size=36, color=GREEN),
            Text("Self-Organized Criticality Emerges", font_size=28, color=YELLOW),
            Text("From Simple Evolutionary Rules", font_size=24, color=GRAY)
        ).arrange(DOWN, buff=0.3)
        summary.fix_in_frame()
        
        # Fade out graph and counters
        self.play(
            FadeOut(self.graph_axes),
            FadeOut(self.graph_title),
            FadeOut(self.graph_labels),
            FadeOut(self.fitness_chart),
            run_time=1
        )
        
        # Rotate quickly to show full 3D structure
        self.play(
            Rotate(
                VGroup(
                    self.species_spheres,
                    self.fitness_bars,
                    self.circle_guide
                ),
                angle=TAU,
                axis=OUT,
                run_time=4,
                rate_func=smooth
            )
        )
        
        # Show summary
        self.play(
            FadeOut(self.title_group),
            FadeIn(summary, shift=UP),
            run_time=2
        )
        
        self.wait(3)
        
        # Final fadeout
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=2
        )


class BakSneppenHistogram(Scene):
    """
    Supplementary scene showing fitness distribution evolution over time.
    """
    
    def construct(self):
        title = Text("Fitness Distribution Evolution", font_size=42, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Create histogram animation
        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 15, 5],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": True},
        )
        axes.to_edge(DOWN, buff=1)
        
        x_label = axes.get_x_axis_label("Fitness", direction=DOWN)
        y_label = axes.get_y_axis_label("Frequency", direction=LEFT)
        
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            run_time=2
        )
        
        # Simulate data and create histogram
        num_bins = 10
        iterations = 30
        num_species = 50
        
        for iteration in range(iterations):
            # Generate fitness distribution
            fitness_values = [random.random() for _ in range(num_species)]
            
            # Find minimum and replace with neighbors
            for _ in range(iteration + 1):
                min_idx = fitness_values.index(min(fitness_values))
                fitness_values[min_idx] = random.random()
                if min_idx > 0:
                    fitness_values[min_idx - 1] = random.random()
                if min_idx < len(fitness_values) - 1:
                    fitness_values[min_idx + 1] = random.random()
            
            # Create histogram bars
            hist, bin_edges = np.histogram(fitness_values, bins=num_bins, range=(0, 1))
            
            bars = VGroup()
            for i in range(num_bins):
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                bar_height = hist[i]
                
                if bar_height > 0:
                    bar = axes.plot_line_graph(
                        x_values=[bin_edges[i], bin_edges[i + 1]],
                        y_values=[bar_height, bar_height],
                        add_vertex_dots=False,
                        line_color=interpolate_color(RED, GREEN, bin_center),
                        stroke_width=20,
                    )
                    bars.add(bar)
            
            if iteration == 0:
                self.play(Create(bars), run_time=1)
                prev_bars = bars
            else:
                self.play(Transform(prev_bars, bars), run_time=0.3)
            
            self.wait(0.2)
        
        self.wait(2)


class BakSneppenAvalanche(ThreeDScene):
    """
    Close-up visualization of a single avalanche cascade event.
    """
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        
        title = Text("Evolutionary Avalanche", font_size=48, color=RED)
        title.fix_in_frame()
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Create small network
        num_species = 12
        radius = 3
        
        fitness_values = [random.random() for _ in range(num_species)]
        spheres = VGroup()
        
        for i in range(num_species):
            angle = i * TAU / num_species
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            sphere = Sphere(center=[x, y, 0], radius=0.3, resolution=(20, 20))
            fitness = fitness_values[i]
            sphere.set_color(interpolate_color(RED, GREEN, fitness))
            sphere.set_sheen(0.5)
            
            spheres.add(sphere)
        
        self.play(LaggedStart(*[GrowFromCenter(s) for s in spheres], lag_ratio=0.1))
        
        # Animate cascading avalanche
        min_idx = fitness_values.index(min(fitness_values))
        
        # Wave of replacements
        cascade_order = [min_idx]
        cascade_order.append((min_idx - 1) % num_species)
        cascade_order.append((min_idx + 1) % num_species)
        
        for idx in cascade_order:
            # Explosion effect
            explosion = Sphere(center=spheres[idx].get_center(), radius=0.3, resolution=(10, 10))
            explosion.set_color(WHITE)
            explosion.set_opacity(0.5)
            
            self.play(
                spheres[idx].animate.scale(0.3).set_opacity(0.2),
                explosion.animate.scale(3).set_opacity(0),
                run_time=0.5
            )
            
            # Regenerate
            new_fitness = random.random()
            new_color = interpolate_color(RED, GREEN, new_fitness)
            
            self.play(
                spheres[idx].animate.scale(1 / 0.3).set_color(new_color).set_opacity(1),
                run_time=0.5
            )
            
            self.wait(0.3)
        
        # Rotate to show final state
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(4)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# Render command suggestions (add to command line):
# manim -pqh bak_sneppen_3d.py BakSneppenEvolution3D
# manim -pqh bak_sneppen_3d.py BakSneppenHistogram
# manim -pqh bak_sneppen_3d.py BakSneppenAvalanche

