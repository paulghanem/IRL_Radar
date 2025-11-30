"""
Quick training script to create a compatible Ant-v4 expert model.
This will train for a reasonable amount of time to get a decent policy.
"""
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

def train_ant_expert(algorithm='PPO', total_timesteps=1_000_000, save_freq=100_000):
    """
    Train an Ant expert using PPO or SAC.

    Args:
        algorithm: 'PPO' or 'SAC'
        total_timesteps: Total training timesteps
        save_freq: How often to save checkpoints
    """
    print("="*60)
    print(f"Training Ant-v4 Expert with {algorithm}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("="*60)

    # Create output directory
    os.makedirs('experts/ant_training', exist_ok=True)

    # Create environment
    env = gym.make('Ant-v4')
    print(f"Environment created: Ant-v4")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Create model
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log='./experts/ant_training/tensorboard/',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log='./experts/ant_training/tensorboard/',
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"\n{algorithm} model created")
    print("Starting training...")
    print("="*60)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./experts/ant_training/checkpoints/',
        name_prefix=f'{algorithm.lower()}_ant'
    )

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )

        # Save final model
        final_path = f'experts/{algorithm.lower()}-Ant-v4-trained.zip'
        model.save(final_path)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Final model saved to: {final_path}")
        print("="*60)

        # Quick test
        print("\nTesting trained model...")
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        print(f"Test reward over 1000 steps: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        model.save(f'experts/{algorithm.lower()}-Ant-v4-interrupted.zip')
        print("Model saved!")

    env.close()
    return model


if __name__ == "__main__":
    # Train with PPO (generally more stable for this task)
    # You can change to 'SAC' if preferred
    # For a quick test, use fewer timesteps (e.g., 100_000)
    # For a good expert, use 1_000_000 or more

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'])
    parser.add_argument('--timesteps', type=int, default=500_000)
    args = parser.parse_args()

    model = train_ant_expert(
        algorithm=args.algo,
        total_timesteps=args.timesteps,
        save_freq=50_000
    )
