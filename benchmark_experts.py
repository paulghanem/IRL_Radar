"""
Benchmark all expert models with multiple seeds.
Tests cumulative rewards over 1000 steps for 10 different seeds.
"""

import os
import sys
import numpy as np
import time
from utils.helpers import GenerateDemo
import json

def benchmark_expert(env_name, max_frames=1000, seeds=None):
    """
    Benchmark expert performance across multiple seeds.

    Args:
        env_name: Environment name
        max_frames: Number of steps per rollout
        seeds: List of seeds to test

    Returns:
        dict: Statistics including mean, std, min, max, and all rewards
    """
    if seeds is None:
        seeds = list(range(10))

    print(f"\n{'='*70}")
    print(f"Benchmarking: {env_name}")
    print(f"{'='*70}")
    print(f"Settings: {max_frames} steps, {len(seeds)} seeds")
    print()

    results = {
        'env_name': env_name,
        'max_frames': max_frames,
        'seeds': seeds,
        'cumulative_rewards': [],
        'final_rewards': [],
        'execution_times': [],
        'state_dims': None,
        'action_dims': None,
    }

    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] Testing seed {seed}...", end=' ', flush=True)

        try:
            start_time = time.time()
            demo_generator = GenerateDemo(env_name, max_frames=max_frames)
            states, actions, cumulative_rewards, env = demo_generator.generate_demo(seed=seed)
            end_time = time.time()

            execution_time = end_time - start_time
            final_reward = cumulative_rewards[-1]

            # Store results
            results['cumulative_rewards'].append(cumulative_rewards.tolist())
            results['final_rewards'].append(float(final_reward))
            results['execution_times'].append(execution_time)

            # Store dimensions (same for all seeds)
            if results['state_dims'] is None:
                results['state_dims'] = states.shape[1]
                results['action_dims'] = actions.shape[1]

            print(f"Final reward: {final_reward:8.2f}, Time: {execution_time:.1f}s")

        except Exception as e:
            print(f"FAILED - {str(e)}")
            results['final_rewards'].append(None)
            results['execution_times'].append(None)
            results['cumulative_rewards'].append(None)

    # Calculate statistics
    valid_rewards = [r for r in results['final_rewards'] if r is not None]

    if valid_rewards:
        results['statistics'] = {
            'mean': float(np.mean(valid_rewards)),
            'std': float(np.std(valid_rewards)),
            'min': float(np.min(valid_rewards)),
            'max': float(np.max(valid_rewards)),
            'median': float(np.median(valid_rewards)),
            'successful_runs': len(valid_rewards),
            'failed_runs': len(results['final_rewards']) - len(valid_rewards),
        }

        print(f"\n{'='*70}")
        print(f"Statistics for {env_name}:")
        print(f"{'='*70}")
        print(f"State dim: {results['state_dims']}, Action dim: {results['action_dims']}")
        print(f"Mean:      {results['statistics']['mean']:8.2f}")
        print(f"Std:       {results['statistics']['std']:8.2f}")
        print(f"Min:       {results['statistics']['min']:8.2f}")
        print(f"Max:       {results['statistics']['max']:8.2f}")
        print(f"Median:    {results['statistics']['median']:8.2f}")
        print(f"Success:   {results['statistics']['successful_runs']}/{len(seeds)}")
        print(f"Avg time:  {np.mean([t for t in results['execution_times'] if t is not None]):.1f}s")
    else:
        results['statistics'] = None
        print(f"\n[ERROR] All runs failed for {env_name}")

    return results


def print_summary_table(all_results):
    """Print a summary table of all benchmarks."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY - EXPERT PERFORMANCE OVER 1000 STEPS")
    print(f"{'='*70}\n")

    # Print table header
    print(f"{'Environment':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Success':<10}")
    print(f"{'-'*70}")

    # Print each environment
    for env_name, results in all_results.items():
        if results['statistics']:
            stats = results['statistics']
            success_rate = f"{stats['successful_runs']}/10"
            print(f"{env_name:<20} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} {stats['max']:<10.2f} {success_rate:<10}")
        else:
            print(f"{env_name:<20} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} {'0/10':<10}")

    print(f"{'-'*70}\n")


def plot_rewards_comparison(all_results, save_path='expert_benchmark_results'):
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        env_names = list(all_results.keys())

        for idx, env_name in enumerate(env_names):
            ax = axes[idx]
            results = all_results[env_name]

            if results['statistics'] and results['cumulative_rewards']:
                # Plot all reward trajectories
                for i, cum_rewards in enumerate(results['cumulative_rewards']):
                    if cum_rewards is not None:
                        ax.plot(cum_rewards, alpha=0.3, linewidth=1)

                # Plot mean trajectory
                valid_cumulative = [r for r in results['cumulative_rewards'] if r is not None]
                if valid_cumulative:
                    mean_trajectory = np.mean(valid_cumulative, axis=0)
                    ax.plot(mean_trajectory, color='red', linewidth=2, label='Mean')

                ax.set_title(f"{env_name} Expert Performance", fontsize=12, fontweight='bold')
                ax.set_xlabel('Steps')
                ax.set_ylabel('Cumulative Reward')
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Add statistics text
                stats = results['statistics']
                stats_text = f"Mean: {stats['mean']:.1f}\nStd: {stats['std']:.1f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        plot_file = os.path.join(save_path, 'expert_performance_comparison.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] Performance plot: {plot_file}")

        plt.close()

        # Create box plot comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        box_data = []
        box_labels = []
        for env_name, results in all_results.items():
            if results['statistics']:
                valid_rewards = [r for r in results['final_rewards'] if r is not None]
                if valid_rewards:
                    box_data.append(valid_rewards)
                    box_labels.append(env_name)

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title('Expert Performance Comparison (1000 steps, 10 seeds)',
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Final Cumulative Reward')
            ax.grid(True, alpha=0.3, axis='y')

            plt.xticks(rotation=15)
            plt.tight_layout()

            box_plot_file = os.path.join(save_path, 'expert_boxplot_comparison.png')
            plt.savefig(box_plot_file, dpi=150, bbox_inches='tight')
            print(f"[SAVED] Box plot: {box_plot_file}")

            plt.close()

        return True

    except ImportError:
        print("\n[INFO] matplotlib not available, skipping plots")
        return False


def main():
    """Main benchmark function."""

    print("\n" + "="*70)
    print("EXPERT BENCHMARK TEST")
    print("="*70)
    print("\nConfiguration:")
    print("  - Rollout length: 1000 steps")
    print("  - Number of seeds: 10")
    print("  - Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print("  - Environments: Walker2d, Hopper, HalfCheetah-v4, Swimmer")
    print("\nEstimated time: ~5-10 minutes")
    print()

    # Define test configuration
    environments = ["Walker2d", "Hopper", "HalfCheetah-v4", "Swimmer"]
    seeds = list(range(10))
    max_frames = 1000

    # Run benchmarks
    all_results = {}
    start_total = time.time()

    for env_name in environments:
        results = benchmark_expert(env_name, max_frames=max_frames, seeds=seeds)
        all_results[env_name] = results

    end_total = time.time()
    total_time = end_total - start_total

    # Print summary
    print_summary_table(all_results)

    # Save results to JSON
    save_path = 'expert_benchmark_results'
    os.makedirs(save_path, exist_ok=True)

    json_file = os.path.join(save_path, 'benchmark_results.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] Results: {json_file}")

    # Save summary statistics to text file
    summary_file = os.path.join(save_path, 'benchmark_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERT BENCHMARK SUMMARY - 1000 STEPS, 10 SEEDS\n")
        f.write("="*70 + "\n\n")

        for env_name, results in all_results.items():
            f.write(f"\n{env_name}:\n")
            f.write(f"  State dim: {results['state_dims']}, Action dim: {results['action_dims']}\n")

            if results['statistics']:
                stats = results['statistics']
                f.write(f"  Mean:      {stats['mean']:8.2f}\n")
                f.write(f"  Std:       {stats['std']:8.2f}\n")
                f.write(f"  Min:       {stats['min']:8.2f}\n")
                f.write(f"  Max:       {stats['max']:8.2f}\n")
                f.write(f"  Median:    {stats['median']:8.2f}\n")
                f.write(f"  Success:   {stats['successful_runs']}/10\n")

                f.write(f"\n  Individual runs:\n")
                for i, (seed, reward) in enumerate(zip(results['seeds'], results['final_rewards'])):
                    if reward is not None:
                        f.write(f"    Seed {seed}: {reward:8.2f}\n")
                    else:
                        f.write(f"    Seed {seed}: FAILED\n")
            else:
                f.write("  All runs failed\n")

        f.write(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)\n")

    print(f"[SAVED] Summary: {summary_file}")

    # Generate plots
    plot_rewards_comparison(all_results, save_path)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {save_path}/")
    print()


if __name__ == "__main__":
    main()
