#!/usr/bin/env python3
"""
Plot IRL experiment results comparing different methods across environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
ENVS = ['Walker2d', 'Hopper']
METHODS = {
    'RGCL (Q=1e-6)': ('rgcl', '1e-06'),
    'RGCL (Q=1e-5)': ('rgcl', '1e-05'),
    'AIRL': ('airl', '1e-05'),
    'GAIL': ('gail', '1e-05'),
    'SQIL': ('sqil', '1e-05'),
    'UB': ('UB', '1e-05'),
    'GCL (baseline)': ('gcl', '1e-05'),
}
SEEDS = [123, 124, 125, 126]
COLORS = {
    'RGCL (Q=1e-6)': '#1f77b4',
    'RGCL (Q=1e-5)': '#ff7f0e',
    'AIRL': '#2ca02c',
    'GAIL': '#d62728',
    'SQIL': '#9467bd',
    'UB': '#8c564b',
    'GCL (baseline)': '#e377c2',
}

def load_cost_data(env, method_dir, Q_value, seed):
    """Load all cost files for a given configuration."""
    costs = []
    iterations = []

    for i in range(10, 1000):  # Check up to iteration 10000
        cost_file = f'results/{env}/{method_dir}/cost_{i*10}_seed={seed}_lambda=0.01_horizon=20_trajectories=500_Q={Q_value}_P=0.01_ndim=16.npy'
        if os.path.exists(cost_file):
            try:
                cost = np.load(cost_file)
                costs.append(cost[-1])
                iterations.append(i*10)
            except Exception as e:
                print(f"Error loading {cost_file}: {e}")
                break
        else:
            break

    return np.array(iterations), np.array(costs)

def plot_learning_curves(env, save_dir='plots'):
    """Plot learning curves for all methods in an environment."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for method_name, (method_dir, Q_value) in METHODS.items():
        all_iterations = []
        all_costs_per_seed = []

        # Load data for all seeds
        for seed in SEEDS:
            iterations, costs = load_cost_data(env, method_dir, Q_value, seed)
            if len(iterations) > 0:
                all_iterations.append(iterations)
                all_costs_per_seed.append(costs)

        if not all_costs_per_seed:
            continue

        # Find common iteration range
        min_len = min([len(c) for c in all_costs_per_seed])
        if min_len == 0:
            continue

        # Truncate all to same length
        iterations_common = all_iterations[0][:min_len]
        costs_matrix = np.array([c[:min_len] for c in all_costs_per_seed])

        # Calculate mean and std
        mean_cost = np.mean(costs_matrix, axis=0)
        std_cost = np.std(costs_matrix, axis=0)

        # Plot
        color = COLORS.get(method_name, None)
        ax.plot(iterations_common, mean_cost, label=method_name, linewidth=2, color=color)
        ax.fill_between(iterations_common, mean_cost - std_cost, mean_cost + std_cost,
                        alpha=0.2, color=color)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Cost (Higher is Better)', fontsize=14)
    ax.set_title(f'{env} - Learning Curves (Mean ± Std across {len(SEEDS)} seeds)', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{env}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{env}_learning_curves.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/{env}_learning_curves.png")
    plt.close()

def plot_final_performance(envs, save_dir='plots', last_n_iters=5):
    """Plot final performance comparison across methods."""
    os.makedirs(save_dir, exist_ok=True)

    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(8*n_envs, 6))
    if n_envs == 1:
        axes = [axes]

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        method_names = []
        method_means = []
        method_stds = []

        for method_name, (method_dir, Q_value) in METHODS.items():
            seed_averages = []

            for seed in SEEDS:
                iterations, costs = load_cost_data(env, method_dir, Q_value, seed)
                if len(costs) >= last_n_iters:
                    # Average last N iterations
                    last_costs = costs[-last_n_iters:]
                    seed_averages.append(np.mean(last_costs))

            if seed_averages:
                method_names.append(method_name)
                method_means.append(np.mean(seed_averages))
                method_stds.append(np.std(seed_averages))

        # Sort by mean performance
        sorted_indices = np.argsort(method_means)[::-1]  # Descending order
        method_names = [method_names[i] for i in sorted_indices]
        method_means = [method_means[i] for i in sorted_indices]
        method_stds = [method_stds[i] for i in sorted_indices]

        # Plot bar chart
        x_pos = np.arange(len(method_names))
        colors_sorted = [COLORS.get(name, 'gray') for name in method_names]

        bars = ax.bar(x_pos, method_means, yerr=method_stds,
                     color=colors_sorted, alpha=0.7, capsize=5)

        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Average Cost (Higher is Better)', fontsize=12)
        ax.set_title(f'{env}\nFinal Performance (Last {last_n_iters} Iterations)', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(method_means, method_stds)):
            ax.text(i, mean + std + (max(method_means) - min(method_means))*0.02,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/final_performance_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/final_performance_comparison.png")
    plt.close()

def plot_seed_comparison(env, method_name, method_dir, Q_value, save_dir='plots'):
    """Plot individual seed trajectories for a specific method."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for seed in SEEDS:
        iterations, costs = load_cost_data(env, method_dir, Q_value, seed)
        if len(iterations) > 0:
            ax.plot(iterations, costs, label=f'Seed {seed}', linewidth=1.5, marker='o',
                   markersize=3, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Cost (Higher is Better)', fontsize=14)
    ax.set_title(f'{env} - {method_name}\nIndividual Seed Trajectories', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/{env}_{method_dir}_Q{Q_value}_seeds.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def print_summary_table(envs):
    """Print a summary table of final performance."""
    print("\n" + "="*100)
    print("SUMMARY TABLE - Average of Last 5 Iterations")
    print("="*100)

    for env in envs:
        print(f"\n{env}:")
        print(f"{'Method':<25} {'Mean':>10} {'Std':>10} {'Seeds':>8}")
        print("-"*60)

        results = []
        for method_name, (method_dir, Q_value) in METHODS.items():
            seed_averages = []

            for seed in SEEDS:
                iterations, costs = load_cost_data(env, method_dir, Q_value, seed)
                if len(costs) >= 5:
                    seed_averages.append(np.mean(costs[-5:]))

            if seed_averages:
                mean = np.mean(seed_averages)
                std = np.std(seed_averages)
                results.append((method_name, mean, std, len(seed_averages)))

        # Sort by mean
        results.sort(key=lambda x: x[1], reverse=True)

        for i, (method, mean, std, n_seeds) in enumerate(results, 1):
            print(f"{i}. {method:<22} {mean:>10.2f} {std:>10.2f} {n_seeds:>8}")

if __name__ == '__main__':
    print("Generating plots for IRL experiment results...")
    print(f"Environments: {ENVS}")
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Seeds: {SEEDS}\n")

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Plot learning curves for each environment
    for env in ENVS:
        print(f"\nPlotting learning curves for {env}...")
        plot_learning_curves(env)

    # Plot final performance comparison
    print("\nPlotting final performance comparison...")
    plot_final_performance(ENVS)

    # Plot seed comparisons for RGCL
    print("\nPlotting seed comparisons for RGCL...")
    for env in ENVS:
        plot_seed_comparison(env, 'RGCL (Q=1e-6)', 'rgcl', '1e-06')
        plot_seed_comparison(env, 'RGCL (Q=1e-5)', 'rgcl', '1e-05')

    # Print summary table
    print_summary_table(ENVS)

    print("\n" + "="*100)
    print("All plots saved in 'plots/' directory")
    print("="*100)
