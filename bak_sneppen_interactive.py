"""
Interactive Bak-Sneppen Exploration
====================================

This script provides an interactive way to explore the Bak-Sneppen model
with real-time parameter adjustments and live visualization updates.

Usage:
    python bak_sneppen_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib import cm
import random


class InteractiveBakSneppen:
    """Interactive Bak-Sneppen model simulator with live plotting."""
    
    def __init__(self, num_species=30, radius=1.0):
        """
        Initialize the interactive simulator.
        
        Args:
            num_species: Number of species in the circle
            radius: Radius of the circular arrangement
        """
        self.num_species = num_species
        self.radius = radius
        self.fitness_values = [random.random() for _ in range(num_species)]
        self.iteration = 0
        
        # Track statistics
        self.min_fitness_history = []
        self.mean_fitness_history = []
        self.replaced_species_history = []
        
        # Setup plot
        self.setup_plot()
    
    def setup_plot(self):
        """Create the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main circular visualization
        self.ax_circle = plt.subplot(2, 3, (1, 4))
        self.ax_circle.set_xlim(-1.5, 1.5)
        self.ax_circle.set_ylim(-1.5, 1.5)
        self.ax_circle.set_aspect('equal')
        self.ax_circle.set_title('Bak-Sneppen Model - Species Circle', fontsize=14, fontweight='bold')
        self.ax_circle.axis('off')
        
        # Fitness evolution plot
        self.ax_fitness = plt.subplot(2, 3, 2)
        self.ax_fitness.set_xlabel('Iteration')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_title('Fitness Evolution', fontweight='bold')
        self.ax_fitness.grid(True, alpha=0.3)
        
        # Fitness distribution histogram
        self.ax_hist = plt.subplot(2, 3, 3)
        self.ax_hist.set_xlabel('Fitness')
        self.ax_hist.set_ylabel('Frequency')
        self.ax_hist.set_title('Fitness Distribution', fontweight='bold')
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.grid(True, alpha=0.3, axis='y')
        
        # Species replacement heatmap
        self.ax_heatmap = plt.subplot(2, 3, 5)
        self.ax_heatmap.set_xlabel('Species ID')
        self.ax_heatmap.set_ylabel('Iteration')
        self.ax_heatmap.set_title('Replacement Heatmap', fontweight='bold')
        
        # Statistics panel
        self.ax_stats = plt.subplot(2, 3, 6)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics', fontweight='bold')
        
        plt.tight_layout()
    
    def fitness_to_color(self, fitness):
        """Convert fitness value to RGB color."""
        cmap = cm.get_cmap('RdYlGn')
        return cmap(fitness)
    
    def update_circle_plot(self):
        """Update the circular species visualization."""
        self.ax_circle.clear()
        self.ax_circle.set_xlim(-1.5, 1.5)
        self.ax_circle.set_ylim(-1.5, 1.5)
        self.ax_circle.set_aspect('equal')
        self.ax_circle.set_title(
            f'Bak-Sneppen Model - Iteration {self.iteration}',
            fontsize=14,
            fontweight='bold'
        )
        self.ax_circle.axis('off')
        
        # Draw circle guide
        circle_guide = Circle((0, 0), self.radius, fill=False, 
                             edgecolor='gray', linestyle='--', linewidth=1, alpha=0.3)
        self.ax_circle.add_patch(circle_guide)
        
        # Draw species
        for i in range(self.num_species):
            angle = i * 2 * np.pi / self.num_species
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            
            # Species circle
            color = self.fitness_to_color(self.fitness_values[i])
            species_circle = Circle((x, y), 0.08, color=color, zorder=10)
            self.ax_circle.add_patch(species_circle)
            
            # Fitness bar
            bar_length = self.fitness_values[i] * 0.4
            bar_x = [x, x + bar_length * np.cos(angle)]
            bar_y = [y, y + bar_length * np.sin(angle)]
            self.ax_circle.plot(bar_x, bar_y, color=color, linewidth=3, alpha=0.7)
            
            # Species ID
            label_x = x * 1.2
            label_y = y * 1.2
            self.ax_circle.text(label_x, label_y, str(i), 
                               ha='center', va='center', fontsize=8, color='black')
        
        # Color bar legend
        fitness_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        legend_elements = [
            mpatches.Patch(color=self.fitness_to_color(f), 
                          label=f'Fitness: {f:.2f}')
            for f in fitness_levels
        ]
        self.ax_circle.legend(handles=legend_elements, loc='upper right', 
                             fontsize=8, framealpha=0.9)
    
    def update_fitness_plot(self):
        """Update fitness evolution plot."""
        self.ax_fitness.clear()
        self.ax_fitness.set_xlabel('Iteration')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_title('Fitness Evolution', fontweight='bold')
        self.ax_fitness.grid(True, alpha=0.3)
        
        if len(self.min_fitness_history) > 0:
            iterations = range(len(self.min_fitness_history))
            self.ax_fitness.plot(iterations, self.min_fitness_history, 
                                'r-', label='Min Fitness', linewidth=2)
            self.ax_fitness.plot(iterations, self.mean_fitness_history, 
                                'g-', label='Mean Fitness', linewidth=2, alpha=0.7)
            
            # Critical threshold line
            self.ax_fitness.axhline(0.667, color='blue', linestyle='--', 
                                   linewidth=1, alpha=0.5, label='Critical Threshold')
            
            self.ax_fitness.legend(fontsize=9)
    
    def update_histogram(self):
        """Update fitness distribution histogram."""
        self.ax_hist.clear()
        self.ax_hist.set_xlabel('Fitness')
        self.ax_hist.set_ylabel('Frequency')
        self.ax_hist.set_title('Fitness Distribution', fontweight='bold')
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.grid(True, alpha=0.3, axis='y')
        
        colors = [self.fitness_to_color(f) for f in self.fitness_values]
        self.ax_hist.hist(self.fitness_values, bins=15, color='steelblue', 
                         alpha=0.7, edgecolor='black')
    
    def update_heatmap(self):
        """Update species replacement heatmap."""
        if len(self.replaced_species_history) > 10:
            # Create heatmap data
            heatmap_data = np.zeros((len(self.replaced_species_history), self.num_species))
            
            for iteration, replaced_indices in enumerate(self.replaced_species_history):
                for idx in replaced_indices:
                    heatmap_data[iteration, idx] = 1
            
            self.ax_heatmap.clear()
            self.ax_heatmap.set_xlabel('Species ID')
            self.ax_heatmap.set_ylabel('Iteration')
            self.ax_heatmap.set_title('Replacement Heatmap (Recent)', fontweight='bold')
            
            # Show only last 50 iterations
            display_data = heatmap_data[-50:] if len(heatmap_data) > 50 else heatmap_data
            
            im = self.ax_heatmap.imshow(display_data, aspect='auto', cmap='Reds', 
                                        interpolation='nearest')
            
            # Colorbar
            if not hasattr(self, 'cbar'):
                self.cbar = plt.colorbar(im, ax=self.ax_heatmap, fraction=0.046)
                self.cbar.set_label('Replaced', rotation=270, labelpad=15)
    
    def update_statistics_panel(self):
        """Update statistics text panel."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics', fontweight='bold')
        
        current_min = min(self.fitness_values)
        current_mean = np.mean(self.fitness_values)
        current_std = np.std(self.fitness_values)
        
        stats_text = f"""
Iteration: {self.iteration}

Current State:
  Min Fitness: {current_min:.4f}
  Mean Fitness: {current_mean:.4f}
  Std Dev: {current_std:.4f}

Historical:
  Total Replacements: {sum(len(r) for r in self.replaced_species_history)}
  Avg Replacements/Iter: {np.mean([len(r) for r in self.replaced_species_history]) if self.replaced_species_history else 0:.2f}

Criticality:
  Distance to Critical: {abs(current_min - 0.667):.4f}
  Progress: {(current_min / 0.667) * 100:.1f}%
        """
        
        self.ax_stats.text(0.1, 0.9, stats_text.strip(), 
                          transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top',
                          family='monospace')
    
    def step(self):
        """Perform one iteration of the Bak-Sneppen model."""
        # Find minimum fitness
        min_fitness = min(self.fitness_values)
        min_idx = self.fitness_values.index(min_fitness)
        
        # Determine neighbors
        left_neighbor = (min_idx - 1) % self.num_species
        right_neighbor = (min_idx + 1) % self.num_species
        
        affected_indices = [left_neighbor, min_idx, right_neighbor]
        
        # Replace with new random values
        for idx in affected_indices:
            self.fitness_values[idx] = random.random()
        
        # Track statistics
        self.min_fitness_history.append(min(self.fitness_values))
        self.mean_fitness_history.append(np.mean(self.fitness_values))
        self.replaced_species_history.append(affected_indices)
        
        self.iteration += 1
    
    def update_all_plots(self):
        """Update all visualization components."""
        self.update_circle_plot()
        self.update_fitness_plot()
        self.update_histogram()
        self.update_heatmap()
        self.update_statistics_panel()
        plt.tight_layout()
    
    def run_interactive(self, num_iterations=100, interval=200):
        """
        Run interactive animation.
        
        Args:
            num_iterations: Number of iterations to run
            interval: Delay between frames in milliseconds
        """
        def animate(frame):
            self.step()
            self.update_all_plots()
            return []
        
        anim = FuncAnimation(self.fig, animate, frames=num_iterations,
                            interval=interval, blit=False, repeat=False)
        
        plt.show()
        
        return anim
    
    def run_manual(self, steps_per_click=1):
        """
        Run with manual stepping (click to advance).
        
        Args:
            steps_per_click: Number of steps to advance per click
        """
        def on_key(event):
            if event.key == ' ':  # Spacebar
                for _ in range(steps_per_click):
                    self.step()
                self.update_all_plots()
                plt.draw()
            elif event.key == 'q':
                plt.close()
        
        self.update_all_plots()
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("\nInteractive Mode:")
        print("  Press SPACE to advance")
        print("  Press Q to quit\n")
        
        plt.show()


def demo_basic():
    """Run basic automated demo."""
    print("Running basic Bak-Sneppen simulation...")
    sim = InteractiveBakSneppen(num_species=30)
    sim.run_interactive(num_iterations=100, interval=100)


def demo_large_system():
    """Demo with larger system."""
    print("Running large-scale Bak-Sneppen simulation...")
    sim = InteractiveBakSneppen(num_species=100)
    sim.run_interactive(num_iterations=200, interval=50)


def demo_manual():
    """Run manual stepping demo."""
    print("Running manual stepping mode...")
    sim = InteractiveBakSneppen(num_species=30)
    sim.run_manual(steps_per_click=1)


def demo_comparison():
    """Compare different system sizes side by side."""
    print("Comparing different system sizes...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sizes = [10, 30, 100]
    iterations = 150
    
    for idx, size in enumerate(sizes):
        print(f"Simulating {size} species...")
        
        sim = InteractiveBakSneppen(num_species=size)
        
        # Run simulation
        for _ in range(iterations):
            sim.step()
        
        # Plot results
        ax = axes[idx]
        ax.plot(sim.min_fitness_history, 'r-', label='Min Fitness', linewidth=2)
        ax.plot(sim.mean_fitness_history, 'g-', label='Mean Fitness', linewidth=2, alpha=0.7)
        ax.axhline(0.667, color='blue', linestyle='--', linewidth=1, 
                  alpha=0.5, label='Critical')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        ax.set_title(f'{size} Species', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("INTERACTIVE BAK-SNEPPEN EXPLORATION")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Available modes:")
        print("  python bak_sneppen_interactive.py basic      # Automated demo")
        print("  python bak_sneppen_interactive.py large      # Large system")
        print("  python bak_sneppen_interactive.py manual     # Manual stepping")
        print("  python bak_sneppen_interactive.py compare    # Size comparison")
        print()
        mode = input("Select mode (or press Enter for basic): ").lower() or 'basic'
    
    if mode == 'basic':
        demo_basic()
    elif mode == 'large':
        demo_large_system()
    elif mode == 'manual':
        demo_manual()
    elif mode == 'compare':
        demo_comparison()
    else:
        print(f"Unknown mode: {mode}")
        demo_basic()

