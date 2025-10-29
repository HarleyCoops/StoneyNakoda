"""
Bak-Sneppen Data Analysis Script
=================================

Analyze simulation data exported from the enhanced Bak-Sneppen visualization.
Generates plots and statistical summaries of evolutionary dynamics.

Usage:
    python analyze_bak_sneppen_data.py bak_sneppen_data_20250129_143022.json
    python analyze_bak_sneppen_data.py --latest
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime


class BakSneppenAnalyzer:
    """Analyzer for Bak-Sneppen simulation data."""
    
    def __init__(self, data_file):
        """
        Initialize analyzer with data file.
        
        Args:
            data_file: Path to JSON data file
        """
        self.data_file = Path(data_file)
        self.load_data()
    
    def load_data(self):
        """Load simulation data from JSON file."""
        print(f"\nLoading data from: {self.data_file}")
        
        with open(self.data_file, 'r') as f:
            self.data = json.load(f)
        
        self.config = self.data['config']
        self.iterations = self.data['iterations']
        self.stats = self.data['statistics']
        
        print(f"✓ Loaded {len(self.iterations)} iterations")
        print(f"  Config: {self.config['num_species']} species, "
              f"{self.config['num_iterations']} iterations")
    
    def plot_fitness_evolution(self, save=True):
        """Plot minimum and mean fitness over time."""
        iterations = range(len(self.stats['min_fitness_evolution']))
        min_fitness = self.stats['min_fitness_evolution']
        mean_fitness = self.stats['mean_fitness_evolution']
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(iterations, min_fitness, 'r-', label='Minimum Fitness', linewidth=2)
        plt.plot(iterations, mean_fitness, 'g-', label='Mean Fitness', linewidth=2, alpha=0.7)
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.title('Fitness Evolution in Bak-Sneppen Model', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            output_file = self.data_file.parent / f"{self.data_file.stem}_fitness_evolution.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_file}")
        
        plt.show()
    
    def plot_fitness_distribution(self, iteration_samples=5, save=True):
        """
        Plot fitness distribution at different time points.
        
        Args:
            iteration_samples: Number of time points to sample
            save: Whether to save the plot
        """
        total_iterations = len(self.iterations)
        sample_indices = np.linspace(0, total_iterations - 1, iteration_samples, dtype=int)
        
        fig, axes = plt.subplots(1, iteration_samples, figsize=(16, 4))
        
        for i, idx in enumerate(sample_indices):
            fitness_values = self.iterations[idx]['fitness_snapshot']
            
            axes[i].hist(fitness_values, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Iteration {idx}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Fitness', fontsize=10)
            axes[i].set_ylabel('Frequency' if i == 0 else '', fontsize=10)
            axes[i].set_ylim([0, max(np.histogram(fitness_values, bins=15)[0]) * 1.2])
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Fitness Distribution Evolution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            output_file = self.data_file.parent / f"{self.data_file.stem}_fitness_dist.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_file}")
        
        plt.show()
    
    def plot_avalanche_statistics(self, save=True):
        """Plot avalanche size statistics."""
        avalanche_sizes = self.stats['avalanche_sizes']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Avalanche size over time
        ax1.plot(avalanche_sizes, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(np.mean(avalanche_sizes), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(avalanche_sizes):.2f}', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Avalanche Size', fontsize=12)
        ax1.set_title('Avalanche Size Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Avalanche size distribution
        unique_sizes, counts = np.unique(avalanche_sizes, return_counts=True)
        ax2.bar(unique_sizes, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Avalanche Size', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Avalanche Size Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            output_file = self.data_file.parent / f"{self.data_file.stem}_avalanches.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_file}")
        
        plt.show()
    
    def plot_species_fitness_trajectories(self, num_species_to_plot=10, save=True):
        """
        Plot fitness trajectories for selected species.
        
        Args:
            num_species_to_plot: Number of random species to track
            save: Whether to save the plot
        """
        num_species = self.config['num_species']
        num_iterations = len(self.iterations)
        
        # Randomly select species to track
        selected_species = np.random.choice(num_species, num_species_to_plot, replace=False)
        
        plt.figure(figsize=(14, 7))
        
        for species_id in selected_species:
            trajectory = []
            for iteration_data in self.iterations:
                trajectory.append(iteration_data['fitness_snapshot'][species_id])
            
            plt.plot(trajectory, alpha=0.6, linewidth=1.5, label=f'Species {species_id}')
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.title(f'Fitness Trajectories for {num_species_to_plot} Random Species',
                  fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            output_file = self.data_file.parent / f"{self.data_file.stem}_trajectories.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_file}")
        
        plt.show()
    
    def calculate_criticality_metrics(self):
        """Calculate metrics related to self-organized criticality."""
        min_fitness = np.array(self.stats['min_fitness_evolution'])
        mean_fitness = np.array(self.stats['mean_fitness_evolution'])
        
        # Critical threshold (often around 0.667 for Bak-Sneppen)
        critical_threshold = 0.667
        
        # Time to reach near-critical state
        near_critical = np.where(min_fitness > critical_threshold * 0.9)[0]
        time_to_critical = near_critical[0] if len(near_critical) > 0 else len(min_fitness)
        
        # Fitness fluctuations
        min_fitness_std = np.std(min_fitness[-50:])  # Last 50 iterations
        mean_fitness_std = np.std(mean_fitness[-50:])
        
        # Avalanche statistics
        avalanche_sizes = np.array(self.stats['avalanche_sizes'])
        
        metrics = {
            'final_min_fitness': min_fitness[-1],
            'final_mean_fitness': mean_fitness[-1],
            'time_to_critical': time_to_critical,
            'min_fitness_std_late': min_fitness_std,
            'mean_fitness_std_late': mean_fitness_std,
            'mean_avalanche_size': np.mean(avalanche_sizes),
            'max_avalanche_size': np.max(avalanche_sizes),
            'total_replacements': np.sum(avalanche_sizes),
        }
        
        return metrics
    
    def print_summary(self):
        """Print comprehensive summary of simulation results."""
        print("\n" + "="*60)
        print("BAK-SNEPPEN SIMULATION SUMMARY")
        print("="*60)
        
        print(f"\nSimulation Timestamp: {self.data['timestamp']}")
        print(f"\nConfiguration:")
        print(f"  Species: {self.config['num_species']}")
        print(f"  Iterations: {self.config['num_iterations']}")
        
        metrics = self.calculate_criticality_metrics()
        
        print(f"\nFitness Statistics:")
        print(f"  Final Min Fitness: {metrics['final_min_fitness']:.4f}")
        print(f"  Final Mean Fitness: {metrics['final_mean_fitness']:.4f}")
        print(f"  Min Fitness Std Dev (late): {metrics['min_fitness_std_late']:.4f}")
        print(f"  Mean Fitness Std Dev (late): {metrics['mean_fitness_std_late']:.4f}")
        
        print(f"\nCriticality Metrics:")
        print(f"  Iterations to Critical State: {metrics['time_to_critical']}")
        print(f"  Total Replacements: {metrics['total_replacements']}")
        
        print(f"\nAvalanche Statistics:")
        print(f"  Mean Avalanche Size: {metrics['mean_avalanche_size']:.2f}")
        print(f"  Max Avalanche Size: {metrics['max_avalanche_size']}")
        
        print("\n" + "="*60 + "\n")
    
    def export_summary_report(self):
        """Export a text summary report."""
        output_file = self.data_file.parent / f"{self.data_file.stem}_report.txt"
        
        metrics = self.calculate_criticality_metrics()
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BAK-SNEPPEN SIMULATION ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.data_file.name}\n")
            f.write(f"Simulation Timestamp: {self.data['timestamp']}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of Species: {self.config['num_species']}\n")
            f.write(f"Number of Iterations: {self.config['num_iterations']}\n\n")
            
            f.write("FITNESS STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Final Minimum Fitness: {metrics['final_min_fitness']:.4f}\n")
            f.write(f"Final Mean Fitness: {metrics['final_mean_fitness']:.4f}\n")
            f.write(f"Min Fitness Std Dev (late): {metrics['min_fitness_std_late']:.4f}\n")
            f.write(f"Mean Fitness Std Dev (late): {metrics['mean_fitness_std_late']:.4f}\n\n")
            
            f.write("CRITICALITY METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Iterations to Critical State: {metrics['time_to_critical']}\n")
            f.write(f"Total Replacements: {metrics['total_replacements']}\n\n")
            
            f.write("AVALANCHE STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean Avalanche Size: {metrics['mean_avalanche_size']:.2f}\n")
            f.write(f"Max Avalanche Size: {metrics['max_avalanche_size']}\n\n")
            
            f.write("="*60 + "\n")
        
        print(f"✓ Exported report: {output_file}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*60)
        print("RUNNING FULL BAK-SNEPPEN DATA ANALYSIS")
        print("="*60)
        
        self.print_summary()
        
        print("\nGenerating plots...")
        self.plot_fitness_evolution()
        self.plot_fitness_distribution()
        self.plot_avalanche_statistics()
        self.plot_species_fitness_trajectories()
        
        self.export_summary_report()
        
        print("\n✓ Analysis complete!\n")


def find_latest_data_file():
    """Find the most recent Bak-Sneppen data file."""
    data_files = list(Path('.').glob('bak_sneppen_data_*.json'))
    
    if not data_files:
        raise FileNotFoundError("No Bak-Sneppen data files found in current directory")
    
    # Sort by modification time
    latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def main():
    """Main entry point for analysis script."""
    parser = argparse.ArgumentParser(
        description='Analyze Bak-Sneppen simulation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_bak_sneppen_data.py data.json
  python analyze_bak_sneppen_data.py --latest
  python analyze_bak_sneppen_data.py data.json --plot fitness
        """
    )
    
    parser.add_argument('data_file', nargs='?', help='Path to JSON data file')
    parser.add_argument('--latest', action='store_true', 
                        help='Use the most recent data file')
    parser.add_argument('--plot', choices=['fitness', 'distribution', 'avalanche', 
                                           'trajectories', 'all'],
                        default='all', help='Which plots to generate')
    parser.add_argument('--no-save', action='store_true', 
                        help='Do not save plots to file')
    
    args = parser.parse_args()
    
    # Determine which data file to use
    if args.latest:
        data_file = find_latest_data_file()
        print(f"\nUsing latest data file: {data_file}")
    elif args.data_file:
        data_file = Path(args.data_file)
        if not data_file.exists():
            print(f"Error: File not found: {data_file}")
            return 1
    else:
        parser.print_help()
        return 1
    
    # Create analyzer
    analyzer = BakSneppenAnalyzer(data_file)
    
    save_plots = not args.no_save
    
    # Generate requested plots
    if args.plot == 'all':
        analyzer.run_full_analysis()
    else:
        analyzer.print_summary()
        
        if args.plot == 'fitness':
            analyzer.plot_fitness_evolution(save=save_plots)
        elif args.plot == 'distribution':
            analyzer.plot_fitness_distribution(save=save_plots)
        elif args.plot == 'avalanche':
            analyzer.plot_avalanche_statistics(save=save_plots)
        elif args.plot == 'trajectories':
            analyzer.plot_species_fitness_trajectories(save=save_plots)
    
    return 0


if __name__ == '__main__':
    exit(main())

