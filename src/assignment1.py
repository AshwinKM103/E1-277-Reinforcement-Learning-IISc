import numpy as np
import logging
import json
import plotly.graph_objects as go
import plotly.io as pio
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """
    Configuration for Bandit Experiments.

    Holds all the hyperparameters and settings required to run the
    non-stationary bandit experiments.

    Args:
        n_runs (int): Number of independent runs to execute.
        n_steps (int): Number of time steps per run.
        n_arms (int): Number of arms in the bandit.
        alpha (float): Step size for the gradient bandit updates.
        beta (float): Weight for variance-based adaptive baseline.
        window_size (int): Window size for baseline calculation.
        change_steps (int): Number of steps between distribution drifts.
        walk_std (float): Standard deviation of the random walk drift.
        output_dir (str): Directory to save output files.
    """
    n_runs: int = 200
    n_steps: int = 2000
    n_arms: int = 10
    alpha: float = 0.1
    beta: float = 0.0 
    window_size: int = 50
    change_steps: int = 500
    walk_std: float = 0.1
    output_dir: str = "outputs"

class AdaptiveGradientBandit:
    """Gradient Bandit Agent with Adaptive Baseline."""
    
    def __init__(self, n_arms: int, alpha: float, beta: float, window_size: int):
        """
        Initialize the AdaptiveGradientBandit with specific parameters.

        Sets up the preference values, action probabilities, and reward history
        buffers required for the gradient bandit algorithm with an adaptive baseline.

        Args:
            n_arms (int): The number of arms in the bandit problem.
            alpha (float): The step-size parameter (alpha) for preference updates.
            beta (float): The adaptation parameter (beta) for the baseline (0 <= beta <= 1).
                          If 0, the baseline is the simple average reward.
            window_size (int): The window size (W) for the recent variance-adjusted mean.
        """
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        
        # H: Preferences for each action, initialized to 0
        self.H = np.zeros(self.n_arms)
        
        # Store current probabilities (pi), initialized to uniform
        self.pi = np.ones(self.n_arms) / self.n_arms
        
        # For baseline calculation
        self.reward_history = deque(maxlen=self.window_size)
        self.total_reward = 0.0
        self.time_step = 0
    
    def select_action(self) -> int:
        """
        Select an action according to the softmax policy derived from preferences.

        Action probabilities pi[a] are calculated using the softmax distribution over
        preferences H[a]:
            pi[a] = exp(H[a]) / Sum(exp(H[b]))

        Numerical stability is ensured by subtracting the maximum preference value
        before exponentiation.

        Returns:
            int: The index of the selected action (arm).
        """
        # Numerical Stability (Shift-Invariance):
        # Subtract max(H) to prevent overflow in exp().
        exp_H = np.exp(self.H - np.max(self.H))
        self.pi = exp_H / np.sum(exp_H)
        
        return np.random.choice(self.n_arms, p=self.pi)
    
    def update(self, action: int, reward: float) -> None:
        """
        Update action preferences based on the received reward.

        Performs a Gradient Bandit update:
        1. For the selected action A_t:
           H[A_t] = H[A_t] + alpha * (Reward - baseline) * (1 - pi[A_t])
        
        2. For non-selected actions a != A_t:
           H[a] = H[a] - alpha * (Reward - baseline) * pi[a]

        Args:
            action (int): The index of the action A_t that was taken.
            reward (float): The reward R received for the action.
        """
        self.time_step += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        
        baseline = self.get_baseline()
        
        # Gradient Ascent Update:
        # H[a] <- H[a] + alpha * (R - baseline) * (1 - pi[a])  if a == A_t
        # H[a] <- H[a] - alpha * (R - baseline) * pi[a]        if a != A_t
        
        # Vectorized update for all arms (term 2)
        self.H -= self.alpha * (reward - baseline) * self.pi
        
        # Correction for selected arm (term 1)
        self.H[action] += self.alpha * (reward - baseline)
    
    def get_baseline(self) -> float:
        """
        Calculate the adaptive baseline for the preference update.

        The baseline is a linear combination of the global average reward and
        a variance-adjusted recent mean:
            baseline_t = (1 - beta) * avg_reward + beta * recent_variance_adjusted_mean

        Where:
        - avg_reward is the running average of all rewards.
        - recent_variance_adjusted_mean considers the last W rewards with weights
          inversely proportional to their squared deviation from the window mean.

        Returns:
            float: The computed baseline value at step t.
        """
        avg_reward = self.total_reward / self.time_step
        
        if self.beta == 0:
            return avg_reward
            
        recent_rewards = np.array(self.reward_history)
        window_mean = np.mean(recent_rewards)
        
        # Weighted variance adjustment
        squared_devs = (recent_rewards - window_mean) ** 2
        epsilon = 1e-8
        weights = 1.0 / (squared_devs + epsilon)
        
        recent_variance_adjusted_mean = np.sum(weights * recent_rewards) / np.sum(weights)
            
        return (1 - self.beta) * avg_reward + self.beta * recent_variance_adjusted_mean

class NonStationaryBandit:
    """Non-stationary 10-armed bandit environment with random walk drift."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the non-stationary bandit environment.

        Sets up the true means for each arm and initializes the random walk 
        parameters.

        Args:
            config (ExperimentConfig): The configuration object containing 
                                       environment parameters like n_arms, 
                                       change_steps, and walk_std.
        """
        self.n_arms = config.n_arms
        self.change_steps = config.change_steps
        self.walk_std = config.walk_std
        self.true_means = np.zeros(self.n_arms)
        self.time_step = 0
        self.reset()
        
    def reset(self) -> None:
        """
        Reset the environment to its initial state.

        Re-initializes the true means of all arms from a standard normal 
        distribution and resets the time step counter.
        """
        self.true_means = np.random.randn(self.n_arms)
        self.time_step = 0
        
    def step(self, action: int) -> float:
        """
        Simulate one time step in the bandit environment.

        Generates a reward from a normal distribution centered at the true mean 
        of the selected arm. Also applies random walk drift to the true means 
        at specified intervals.

        Args:
            action (int): The index of the arm to pull.
            
        Returns:
            float: The sampled reward.
        """
        self.time_step += 1
        
        # Apply non-stationary drift every `change_steps`
        if (self.time_step > 0) and (self.time_step % self.change_steps == 0):
            logging.debug(f"Applying random walk drift at step {self.time_step}")
            self.true_means += np.random.normal(0, self.walk_std, size=self.n_arms)
            
        return np.random.normal(self.true_means[action], 1.0)
        
    def get_optimal_action(self) -> int:
        """
        Identify the optimal action at the current time step.

        Returns:
            int: The index of the arm with the highest current expected reward.
        """
        return np.argmax(self.true_means)

def setup_experiment_dirs(base_dir: str = "outputs") -> Tuple[Path, Path]:
    """
    Create the directory structure for experiment outputs.

    Creates unique timestamped directories for logs and results to ensure 
    experiments do not overwrite each other. Sets up file-based logging.

    Args:
        base_dir (str): The root directory for all experiment outputs.

    Returns:
        Tuple[Path, Path]: A tuple containing paths to the log directory and 
                           the results directory (log_dir, results_dir).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp
    
    input_logs = exp_dir / "logs"
    results_dir = exp_dir / "results"
    
    input_logs.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure file logging
    log_file = input_logs / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Created output directories at {exp_dir}")
    logger.info(f"Logging to file: {log_file}")
    return input_logs, results_dir

def save_results_to_json(results: Dict, output_dir: Path, filename: str) -> None:
    """
    Save the experiment results to a JSON file.

    Handles serialiation of NumPy arrays to ensure JSON compatibility.

    Args:
        results (Dict): The dictionary containing experiment data.
        output_dir (Path): The directory to save the file in.
        filename (str): The name of the output JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        k: v.tolist() if isinstance(v, np.ndarray) else v 
        for k, v in results.items()
    }
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    logger.info(f"Saved results to {filepath}")

def run_experiment(n_runs: int, n_steps: int, n_arms: int, alpha: float, beta: float, window_size: int) -> Dict[str, np.ndarray]:
    """
    Execute the bandit experiment across multiple runs.

    Runs the simulation for a specified number of steps and runs, collecting 
    metrics such as rewards, optimal action selections, and baseline values.

    Args:
        n_runs (int): The number of independent simulation runs to perform.
        n_steps (int): The number of time steps per run.
        n_arms (int): The number of arms in the bandit problem.
        alpha (float): The step-size parameter for the agent.
        beta (float): The baseline variance weight.
        window_size (int): The window size for the agent's baseline calculation.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing averaged results for 
                               'rewards', 'optimal_actions', 'baselines', 
                               'regrets', plus the configuration used.
    """
    # Reconstruct config for internal use
    config = ExperimentConfig(
        n_runs=n_runs,
        n_steps=n_steps,
        n_arms=n_arms,
        alpha=alpha,
        beta=beta,
        window_size=window_size
    )

    logger.info(f"Starting Experiment: Beta={beta}, Runs={n_runs}, Steps={n_steps}")
    
    # Cumulative stats
    total_rewards = np.zeros(n_steps)
    total_optimal_actions = np.zeros(n_steps)
    total_baselines = np.zeros(n_steps)
    total_regrets = np.zeros(n_steps)
    
    for run in range(n_runs):
        if run % 50 == 0:
             logger.debug(f"Simulating Run {run}/{n_runs}")
             
        env = NonStationaryBandit(config)
        agent = AdaptiveGradientBandit(n_arms, alpha, beta, window_size)
        
        for t in range(n_steps):
            action = agent.select_action()
            reward = env.step(action)
            
            optimal_action = env.get_optimal_action()
            is_optimal = 1 if action == optimal_action else 0
            
            # Calculate instantaneous regret based on true means
            # Regret = Optimal Expected Reward - Selected Expected Reward
            inst_regret = env.true_means[optimal_action] - env.true_means[action]
            
            agent.update(action, reward)
            
            total_rewards[t] += reward
            total_optimal_actions[t] += is_optimal
            total_baselines[t] += agent.get_baseline()
            total_regrets[t] += inst_regret
            
    # Compute Averages
    results = {
        'rewards': total_rewards / n_runs,
        'optimal_actions': (total_optimal_actions / n_runs) * 100,
        'baselines': total_baselines / n_runs,
        'regrets': np.cumsum(total_regrets / n_runs),
        'config': asdict(config)
    }
    return results

def plot_results(all_results: Dict[str, Dict], output_dir: Path) -> None:
    """
    Generate and save Plotly figures for the experiment results.

    Creates line charts for average reward, optimal action percentage, and 
    baseline evolution over time. Images are saved as static files.

    Args:
        all_results (Dict[str, Dict]): A dictionary mapping experiment labels 
                                       to result dictionaries.
        output_dir (Path): The directory where plots will be saved.
    """
    logger.info("Generating plots...")
    
    # Metrics to plot as per assignment requirements
    metrics = [
        ('rewards', 'Average Reward', 'Average Reward vs Time'),
        ('optimal_actions', '% Optimal Action', 'Optimal Action Selection %'),
        ('baselines', 'Baseline Value', 'Adaptive Baseline Evolution'),
        ('regrets', 'Cumulative Regret', 'Cumulative Regret vs Time')
    ]
    
    for key, ylabel, title in metrics:
        fig = go.Figure()
        
        for label, res in all_results.items():
            data = res[key]
            # Simple moving average for smoother visualization
            if key in ['rewards', 'optimal_actions']:
                window = 100
                data = np.convolve(data, np.ones(window)/window, mode='valid')
                x_axis = np.arange(window-1, len(data) + window - 1)
            else:
                x_axis = np.arange(len(data))
                
            fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines', name=label))
            
        fig.update_layout(
            title=title,
            xaxis_title='Steps',
            yaxis_title=ylabel,
            template="plotly_white",
            legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save as PNG
        filename = f"{key}.png"
        try:
            fig.write_image(output_dir / filename)
            logger.info(f"Saved plot: {filename}")
        except Exception as e:
            logger.error(f"Failed to save static image for {key}. Is kaleido installed? Error: {e}")

def main():
    """Main execution entry point."""
    np.random.seed(42)
    log_dir, results_dir = setup_experiment_dirs()
    
    # Configurations to test
    betas = [0.0, 0.3, 0.6]
    all_results = {}
    
    base_config = ExperimentConfig() # Use defaults
    
    for beta in betas:
        # Create specific config for this beta
        config = ExperimentConfig(
            n_runs=base_config.n_runs,
            n_steps=base_config.n_steps,
            beta=beta,
            output_dir=str(results_dir)
        )
        
        label = f"Beta = {beta}"
        if beta == 0.0:
            label += " (Standard)"
        
        results = run_experiment(
            n_runs=config.n_runs,
            n_steps=config.n_steps,
            n_arms=config.n_arms,
            alpha=config.alpha,
            beta=config.beta,
            window_size=config.window_size
        )
        all_results[label] = results
        
        # Save individual raw data
        save_results_to_json(results, results_dir, f"results_beta_{beta}.json")
        
    # Generate Plots
    plot_results(all_results, results_dir)
    logger.info("Experiment sequence completed successfully.")

if __name__ == "__main__":
    main()
