"""
assignment2.py

Implements Q-Learning on a 5×5 Tabular GridWorld

The implementation supports:
    - Customisable N×N grid environment supporting various cell types.
    - Off-policy TD-control agent with epsilon-greedy exploration.
    - Episodic training loop that drives agent-environment interaction.
    - Plotly-based policy and value-function visualization.

Key classes / functions:
    - GridWorld: A customisable N×N grid environment.
    - QLearningAgent: Off-policy TD-control agent.
    - train: Episodic training loop.
    - check_convergence: Rolling-average stability check for convergence detection.

Version:
    - 07-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""

import sys
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from loguru import logger
import plotly.graph_objects as go


# ============================================================
#   EXPERIMENT CONFIGURATION
# ============================================================

@dataclass
class ExperimentConfig:
    """
    Hyperparameters and experiment settings for the Q-Learning run.

    Args:
        alpha (float): Learning rate for Q-table updates.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration probability for ε-greedy policy.
        epsilon_min (float): Lower bound on epsilon after decay.
        epsilon_decay (float): Multiplicative decay factor applied per episode.
        n_episodes (int): Total number of training episodes.
        convergence_window (int): Rolling window size used by check_convergence.
        convergence_threshold (float): Std-dev threshold below which training
                                       is considered converged.
        seed (int): Random seed for reproducibility.
        output_dir (str): Root directory where outputs are written.
    """
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99
    n_episodes: int = 500
    convergence_window: int = 50
    convergence_threshold: float = 0.5
    seed: int = 777
    output_dir: str = "outputs/assignment2"


# ============================================================
#   OUTPUT DIRECTORY SETUP
# ============================================================

def setup_experiment_dirs(base_dir: str = "outputs/assignment2") -> Tuple[Path, Path]:
    """
    Create timestamped directory structure for the experiment run.

    Creates the following layout under *base_dir*/<timestamp>/:
        logs/   – log files
        plots/  – visualisation outputs

    Configures Loguru to write to the log file in addition to stderr.

    Args:
        base_dir (str): Root output directory. Defaults to 'outputs/assignment2'.

    Returns:
        Tuple[Path, Path]: (log_dir, plots_dir) – absolute paths to the
                           log and plots sub-directories.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp

    log_dir = exp_dir / "logs"
    plots_dir = exp_dir / "plots"

    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Reset loguru and add stderr + file sinks
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}:{function}:{line}</cyan> — {message}",
    )
    log_file = log_dir / "experiment.log"
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
    )

    logger.info(f"Experiment directory created: {exp_dir}")
    logger.info(f"Log file: {log_file}")
    return log_dir, plots_dir


def save_config(config: ExperimentConfig, exp_dir: Path) -> None:
    """
    Serialise and save the experiment configuration to a JSON file.

    Args:
        config (ExperimentConfig): Dataclass holding all hyperparameters.
        exp_dir (Path): Experiment root directory where config.json is written.
    """
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


# ============================================================
#   GRIDWORLD ENVIRONMENT
# ============================================================

class GridWorld:
    """
    A customisable N × N GridWorld environment.

    Grid cell types:
        'S' – Start state (step reward = −0.1)
        'G' – Goal state  (reward = +10, episode ends)
        'P' – Pit / Trap  (reward = −5,  episode ends)
        'W' – Wall        (impassable; agent stays in place)
        '.' – Empty cell  (reward = −0.1 step cost)

    States are encoded as integers 0 … N*M−1 in row-major order.

    Args:
        grid (List[List[str]]): 2-D list of cell-type characters.
    """

    ACTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    DELTA   = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    # Reward mapping keyed by cell type
    REWARDS = {"G": 10.0, "P": -5.0, "S": -0.1, ".": -0.1, "W": -0.1}

    def __init__(self, grid: List[List[str]]) -> None:
        """
        Initialise the GridWorld from a 2-D character grid.

        Args:
            grid (List[List[str]]): Grid layout. Each inner list is a row.

        Raises:
            ValueError: If 'S' or 'G' characters are absent from *grid*.
        """
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.n_states  = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = self._find("S")
        self.goal_state  = self._find("G")
        self.current_state = self.start_state

    def _find(self, char: str) -> int:
        """
        Locate the first occurrence of *char* in the grid.

        Args:
            char (str): The cell-type character to search for.

        Returns:
            int: State index (row-major) of the matching cell.

        Raises:
            ValueError: If *char* is not found anywhere in the grid.
        """
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == char:
                    return r * self.n_cols + c
        raise ValueError(f"Character '{char}' not found in grid")

    def state_to_rc(self, state: int) -> Tuple[int, int]:
        """
        Convert a flat state index to (row, col) coordinates.

        Args:
            state (int): Flat state index in [0, n_states).

        Returns:
            Tuple[int, int]: (row, col) grid coordinates.
        """
        return state // self.n_cols, state % self.n_cols

    def rc_to_state(self, r: int, c: int) -> int:
        """
        Convert (row, col) coordinates to a flat state index.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            int: Flat state index.
        """
        return r * self.n_cols + c

    def reset(self) -> int:
        """
        Reset the environment to the start state.

        Returns:
            int: The start state index.
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute one step in the environment.

        Transition rules (deterministic):
        - If the resulting cell is outside the grid boundary → stay.
        - If the resulting cell is a Wall ('W') → stay.
        - Otherwise → move to the new cell.

        Reward is determined by the cell the agent *ends up in* (after
        applying boundary / wall checks).

        The episode terminates (done=True) when the agent enters Goal or Pit.

        Args:
            action (int): Action index — 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT.

        Returns:
            Tuple[int, float, bool]:
                next_state (int)  – flat index of the resulting state.
                reward     (float) – reward signal for this transition.
                done       (bool) – True if the episode has ended.
        """
        r, c = self.state_to_rc(self.current_state)
        dr, dc = self.DELTA[action]
        nr, nc = r + dr, c + dc

        # Boundary and wall checks — agent stays if move is invalid
        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            # Out of bounds → stay in current cell
            nr, nc = r, c
        elif self.grid[nr][nc] == "W":
            # Wall collision → stay in current cell
            nr, nc = r, c

        cell_type = self.grid[nr][nc]
        reward = self.REWARDS.get(cell_type, -0.1)
        done = cell_type in ("G", "P")

        self.current_state = self.rc_to_state(nr, nc)
        return self.current_state, reward, done


# ============================================================
#   Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.

    Implements the off-policy TD-control algorithm:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

    Args:
        n_states (int): Number of states in the environment.
        n_actions (int): Number of actions available.
        alpha (float): Learning rate α ∈ (0, 1]. Default 0.1.
        gamma (float): Discount factor γ ∈ [0, 1]. Default 0.99.
        epsilon (float): Initial exploration probability. Default 1.0.
        epsilon_min (float): Minimum epsilon after decay. Default 0.01.
        epsilon_decay (float): Multiplicative decay per episode. Default 0.995.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table initialised to zeros, shape (|S|, |A|)
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """
        Choose an action using the ε-greedy policy.

        With probability ε a uniformly random action is taken (exploration);
        otherwise the action with the highest Q-value for *state* is chosen
        (exploitation). Ties in Q-values are broken by np.argmax (lowest index).

        Args:
            state (int): Current environment state index.

        Returns:
            int: Selected action index in [0, n_actions).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Apply the Q-learning (off-policy TD) update rule.

        Update formula:
            td_target = r                              if done
                      = r + γ * max_a' Q(s', a')      otherwise
            Q(s,a) ← Q(s,a) + α * (td_target − Q(s,a))

        Args:
            state (int):      Current state index.
            action (int):     Action taken.
            reward (float):   Reward received after the action.
            next_state (int): Resulting state index.
            done (bool):      True if the episode terminated after this step.
        """
        if done:
            # Terminal transition: no bootstrapping from next state
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """
        Apply exponential decay to the exploration parameter ε.

        ε ← max(ε_min, ε × ε_decay)

        Called once per episode, outside the inner step loop.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
#   TRAINING LOOP
# ============================================================

def train(env: GridWorld, agent: QLearningAgent, n_episodes: int = 500) -> List[float]:
    """
    Train the Q-Learning agent on the environment for *n_episodes* episodes.

    Each episode:
        1. Resets the environment to obtain the initial state.
        2. Repeats until done:
            a. Agent selects an action (ε-greedy).
            b. Environment returns (next_state, reward, done).
            c. Agent updates the Q-table.
        3. Epsilon is decayed at the end of the episode.
        4. Total episode reward is appended to the history.

    Args:
        env (GridWorld): The GridWorld environment instance.
        agent (QLearningAgent): The Q-Learning agent instance.
        n_episodes (int): Number of episodes to train for. Default 500.

    Returns:
        List[float]: Total reward obtained during each episode, length n_episodes.
    """
    logger.info(f"Training started — {n_episodes} episodes")
    reward_history: List[float] = []

    for episode in range(n_episodes):
        # Reset environment at the start of each episode
        state = env.reset()
        total_reward = 0.0
        done = False

        # Inner loop: interact until episode terminates
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        # Decay exploration after each full episode
        agent.decay_epsilon()
        reward_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            recent_mean = float(np.mean(reward_history[-50:]))
            logger.info(
                f"Episode {episode + 1:>4}/{n_episodes} | "
                f"Total Reward: {total_reward:>7.2f} | "
                f"Rolling Mean (last 50): {recent_mean:>7.2f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    logger.info("Training complete.")
    return reward_history


# ============================================================
#   CONVERGENCE CHECK
# ============================================================

def check_convergence(
    reward_history: List[float],
    window: int = 50,
    threshold: float = 0.5,
) -> Tuple[bool, List[float]]:
    """
    Assess whether training has converged by measuring stability of the
    rolling average reward.

    Convergence criterion: the standard deviation of the *last* `window`
    rolling-average values is below `threshold`.

    A rolling average is first computed over the full reward history with the
    specified window size. The std-dev of the tail of that series is then
    compared against the threshold.

    Args:
        reward_history (List[float]): Per-episode total rewards from train().
        window (int): Size of the rolling window. Default 50.
        threshold (float): Std-dev threshold for convergence. Default 0.5.

    Returns:
        Tuple[bool, List[float]]:
            converged (bool)          – True if convergence criterion is met.
            rolling_avg (List[float]) – Rolling mean series (length =
                                        len(reward_history) − window + 1).
    """
    rolling_avg: List[float] = []
    for i in range(window - 1, len(reward_history)):
        rolling_avg.append(float(np.mean(reward_history[i - window + 1 : i + 1])))

    if len(rolling_avg) < window:
        logger.warning(
            "Insufficient data: need at least 2×window episodes to assess convergence."
        )
        return False, rolling_avg

    tail_std = float(np.std(rolling_avg[-window:]))
    converged = tail_std < threshold
    logger.info(
        f"Convergence check — tail std-dev: {tail_std:.4f} | "
        f"threshold: {threshold} | converged: {converged}"
    )
    return converged, rolling_avg


# ============================================================
#   VISUALISATION HELPERS
# ============================================================

def _cell_color(cell_type: str) -> str:
    """
    Return an RGBA fill colour string for a given grid cell type.

    Args:
        cell_type (str): One of 'S', 'G', 'P', 'W', '.'.

    Returns:
        str: CSS rgba() colour string.
    """
    palette = {
        "S": "rgba(100, 149, 237, 0.9)",    # Cornflower blue – start
        "G": "rgba( 50, 205,  50, 0.9)",    # Lime green – goal
        "P": "rgba(220,  20,  60, 0.9)",    # Crimson – pit
        "W": "rgba( 60,  60,  60, 0.9)",    # Dark grey – wall
        ".": "rgba(240, 240, 255, 0.9)",    # Pale lavender – empty
    }
    return palette.get(cell_type, "rgba(200,200,200,0.5)")


# ============================================================
#   VISUALISATION – POLICY
# ============================================================

def visualize_policy(
    env: GridWorld,
    agent: QLearningAgent,
    plots_dir: Path = Path("."),
    save: bool = True,
) -> go.Figure:
    """
    Visualise the greedy policy π*(s) = argmax_a Q(s,a) on the grid.

    Each non-wall, non-terminal cell is annotated with a Unicode arrow:
        UP → '▲', DOWN → '▼', LEFT → '◀', RIGHT → '▶'
    Goal, Pit, and Wall cells are shown with distinct background colours.

    The coordinate system uses data coordinates (col=x, row=y) with
    y=0 at the TOP (row 0). Shapes and annotations share the same coordinate
    space, ensuring cells and arrows are perfectly aligned.

    Args:
        env (GridWorld): The trained environment instance.
        agent (QLearningAgent): The trained Q-Learning agent.
        plots_dir (Path): Directory where plot files are written. Default '.'.
        save (bool): Whether to write files to disk. Default True.

    Returns:
        go.Figure: The constructed Plotly figure.
    """
    logger.info("Generating policy visualisation…")

    action_arrows = {0: "▲", 1: "▼", 2: "◀", 3: "▶"}

    nrows = env.n_rows
    ncols = env.n_cols

    fig = go.Figure()
    shapes = []
    annotations = []

    for r in range(nrows):
        for c in range(ncols):
            state = env.rc_to_state(r, c)
            cell  = env.grid[r][c]
            color = _cell_color(cell)
            y_bottom = nrows - 1 - r          # rectangle lower edge
            y_top    = nrows - 1 - r + 1      # rectangle upper edge
            y_center = nrows - 1 - r + 0.5   # annotation centre

            shapes.append(
                dict(
                    type="rect",
                    x0=c, x1=c + 1,
                    y0=y_bottom, y1=y_top,
                    fillcolor=color,
                    line=dict(color="white", width=2),
                )
            )

            # Build annotation text
            if cell == "W":
                text = "■"
                font_color = "white"
                font_size = 22
            elif cell == "G":
                text = "G"
                font_color = "white"
                font_size = 20
            elif cell == "P":
                text = "P"
                font_color = "white"
                font_size = 20
            elif cell == "S":
                action = int(np.argmax(agent.Q[state]))
                text = "S " + action_arrows[action]
                font_color = "black"
                font_size = 15
            else:
                action = int(np.argmax(agent.Q[state]))
                text = action_arrows[action]
                font_color = "#1a1a2e"
                font_size = 22

            annotations.append(
                dict(
                    x=c + 0.5,
                    y=y_center,
                    text=text,
                    showarrow=False,
                    font=dict(size=font_size, color=font_color),
                    xanchor="center",
                    yanchor="middle",
                )
            )

    # Row / column tick labels on the axes
    col_labels = [str(c) for c in range(ncols)]
    row_labels = [str(r) for r in range(nrows)]   # row 0 is at top

    fig.update_layout(
        title=dict(
            text="<b>Greedy Policy  π*(s) = argmax<sub>a</sub> Q(s,a)</b>",
            x=0.5,
            font=dict(size=18, family="Arial"),
        ),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[0, ncols],
            tickvals=[c + 0.5 for c in range(ncols)],
            ticktext=col_labels,
            title="Column",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            # y=nrows-1+0.5 = top → row 0;  y=0+0.5 = bottom → row nrows-1
            range=[0, nrows],
            tickvals=[(nrows - 1 - r) + 0.5 for r in range(nrows)],
            ticktext=row_labels,
            title="Row",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        width=560,
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=70, b=50),
    )

    if save:
        html_path = plots_dir / "policy.html"
        png_path  = plots_dir / "policy.png"
        fig.write_html(str(html_path))
        logger.info(f"Policy plot (HTML) saved to {html_path}")
        try:
            fig.write_image(str(png_path))
            logger.info(f"Policy plot (PNG)  saved to {png_path}")
        except Exception as exc:
            logger.warning(f"Could not save PNG (is kaleido installed?): {exc}")

    return fig


# ============================================================
#   VISUALISATION – VALUE FUNCTION
# ============================================================

def visualize_value_function(
    env: GridWorld,
    agent: QLearningAgent,
    plots_dir: Path = Path("."),
    save: bool = True,
) -> go.Figure:
    """
    Visualise the value function V(s) = max_a Q(s,a) as a colour heatmap.

    Each cell is colour-coded by its value and annotated with the numerical
    value rounded to two decimal places. Wall cells are labelled "Wall".
    Args:
        env (GridWorld): The trained environment instance.
        agent (QLearningAgent): The trained Q-Learning agent.
        plots_dir (Path): Directory where plot files are written. Default '.'.
        save (bool): Whether to write files to disk. Default True.

    Returns:
        go.Figure: The constructed Plotly figure.
    """
    logger.info("Generating value function heatmap…")

    nrows = env.n_rows
    ncols = env.n_cols

    # V(s) = max_a Q(s,a), shaped as a 2-D grid (row-major, row 0 = top)
    V = np.max(agent.Q, axis=1).reshape(nrows, ncols)

    # Plotly Heatmap places z[0] at y=0 (bottom).
    # Flip rows so that grid row 0 appears at the visual top.
    V_display = V[::-1, :]   # shape (nrows, ncols), V_display[0] == grid row (nrows-1)

    # Build annotation text in the same flipped order
    ann_text = []
    for r_display in range(nrows):
        r_grid = nrows - 1 - r_display   # map display row → grid row
        row_text = []
        for c in range(ncols):
            cell = env.grid[r_grid][c]
            if cell == "W":
                row_text.append("Wall")
            elif cell == "G":
                row_text.append(f"G\n{V[r_grid, c]:.2f}")
            elif cell == "P":
                row_text.append(f"P\n{V[r_grid, c]:.2f}")
            elif cell == "S":
                row_text.append(f"S\n{V[r_grid, c]:.2f}")
            else:
                row_text.append(f"{V[r_grid, c]:.2f}")
        ann_text.append(row_text)

    # Y-axis tick labels: y=0 → grid row (nrows-1), y=nrows-1 → grid row 0
    y_tickvals  = list(range(nrows))
    y_ticktext  = [str(nrows - 1 - i) for i in range(nrows)]

    fig = go.Figure(
        data=go.Heatmap(
            z=V_display,
            colorscale="RdYlGn",
            colorbar=dict(title="V(s)", thickness=15, len=0.9),
            text=ann_text,
            texttemplate="%{text}",
            textfont=dict(size=12, color="black"),
            hovertemplate="Row %{y}, Col %{x}<br>V = %{z:.3f}<extra></extra>",
            showscale=True,
            zmin=float(V.min()),
            zmax=float(V.max()),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Value Function  V(s) = max<sub>a</sub> Q(s,a)</b>",
            x=0.5,
            font=dict(size=18, family="Arial"),
        ),
        xaxis=dict(
            tickvals=list(range(ncols)),
            ticktext=[str(c) for c in range(ncols)],
            title="Column",
            side="top",
        ),
        yaxis=dict(
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            title="Row",
        ),
        width=620,
        height=570,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=100, b=40),
    )

    if save:
        html_path = plots_dir / "value_function.html"
        png_path  = plots_dir / "value_function.png"
        fig.write_html(str(html_path))
        logger.info(f"Value function plot (HTML) saved to {html_path}")
        try:
            fig.write_image(str(png_path))
            logger.info(f"Value function plot (PNG)  saved to {png_path}")
        except Exception as exc:
            logger.warning(f"Could not save PNG (is kaleido installed?): {exc}")

    return fig


# ============================================================
#   VISUALISATION – LEARNING CURVE  (Part C.2)
# ============================================================

def plot_learning_curve(
    reward_history: List[float],
    rolling_avg: List[float],
    window: int = 50,
    plots_dir: Path = Path("."),
    save: bool = True,
) -> go.Figure:
    """
    Plot the raw episode rewards and rolling average on a single figure.

    Two traces are drawn:
        1. Raw episode rewards – very faint background, shows the episode-to-episode
           variance and the initial noisy exploration phase.
        2. Rolling average (window=`window`) – bold solid line, clearly shows
           the training trend and the approximate convergence point.

    Args:
        reward_history (List[float]): Raw per-episode total rewards.
        rolling_avg (List[float]): Rolling-average series from check_convergence.
        window (int): Window size used for the rolling average. Default 50.
        plots_dir (Path): Directory where plot files are written. Default '.'.
        save (bool): Whether to write files to disk. Default True.

    Returns:
        go.Figure: The constructed Plotly figure.
    """
    logger.info("Generating learning curve plot…")

    episodes = list(range(1, len(reward_history) + 1))
    # Rolling average series starts at episode `window`
    rolling_episodes = list(range(window, len(reward_history) + 1))

    fig = go.Figure()

    # Trace 1: raw rewards – very faint, thin grey line
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=reward_history,
            mode="lines",
            name="Episode Reward (raw)",
            line=dict(color="rgba(150,150,200,0.3)", width=1),
        )
    )

    # Trace 2: rolling average – prominent solid orange line
    fig.add_trace(
        go.Scatter(
            x=rolling_episodes,
            y=rolling_avg,
            mode="lines",
            name=f"Rolling Avg (w={window})",
            line=dict(color="darkorange", width=2.5),
        )
    )


    fig.update_layout(
        title=dict(
            text="<b>Q-Learning — Training Learning Curve</b>",
            x=0.5,
            font=dict(size=18, family="Arial"),
        ),
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        legend=dict(
            yanchor="bottom", y=0.05,
            xanchor="right", x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        template="plotly_white",
        width=850,
        height=500,
        margin=dict(l=70, r=30, t=80, b=60),
    )

    if save:
        html_path = plots_dir / "learning_curve.html"
        png_path  = plots_dir / "learning_curve.png"
        fig.write_html(str(html_path))
        logger.info(f"Learning curve (HTML) saved to {html_path}")
        try:
            fig.write_image(str(png_path))
            logger.info(f"Learning curve (PNG)  saved to {png_path}")
        except Exception as exc:
            logger.warning(f"Could not save PNG (is kaleido installed?): {exc}")

    return fig


# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Experiment Configuration
    # ------------------------------------------------------------------
    config = ExperimentConfig()
    np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # 2. Setup output directories and logging
    # ------------------------------------------------------------------
    log_dir, plots_dir = setup_experiment_dirs(config.output_dir)
    exp_dir = log_dir.parent
    save_config(config, exp_dir)

    logger.info("=" * 65)
    logger.info("  Q-Learning on a Tabular GridWorld — Assignment 2")
    logger.info("=" * 65)
    logger.info(f"Config: {asdict(config)}")

    # ------------------------------------------------------------------
    # 3. Define the 5×5 GridWorld
    #      S at (row 0, col 0)  →  state 0
    #      G at (row 4, col 4)  →  state 24
    #      P at (row 3, col 2)  →  state 17
    #    Walls at: (0,3),(1,1),(1,3),(2,1),(3,3)
    # ------------------------------------------------------------------
    GRID = [
        ["S", ".", ".", "W", "."],
        [".", "W", ".", "W", "."],
        [".", "W", ".", ".", "."],
        [".", ".", "P", "W", "."],
        [".", ".", ".", ".", "G"],
    ]

    env = GridWorld(GRID)
    logger.info(
        f"GridWorld: {env.n_rows}×{env.n_cols} | "
        f"States={env.n_states} | Actions={env.n_actions} | "
        f"Start={env.start_state} | Goal={env.goal_state}"
    )

    agent = QLearningAgent(
        n_states      = env.n_states,
        n_actions     = env.n_actions,
        alpha         = config.alpha,
        gamma         = config.gamma,
        epsilon       = config.epsilon,
        epsilon_min   = config.epsilon_min,
        epsilon_decay = config.epsilon_decay,
    )
    logger.info(
        f"Agent: α={agent.alpha} | γ={agent.gamma} | "
        f"ε₀={agent.epsilon} | ε_min={agent.epsilon_min} | "
        f"ε_decay={agent.epsilon_decay}"
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    rewards = train(env, agent, n_episodes=config.n_episodes)

    # ------------------------------------------------------------------
    # 5. Convergence analysis
    # ------------------------------------------------------------------
    converged, rolling_avg = check_convergence(
        rewards,
        window=config.convergence_window,
        threshold=config.convergence_threshold,
    )
    logger.info(f"Converged: {converged}")
    print(f"Converged: {converged}")

    # Log final greedy path for verification
    env.reset()
    greedy_path = []
    action_symbols = {0: "^", 1: "v", 2: "<", 3: ">"}
    for _ in range(30):
        s = env.current_state
        r0, c0 = env.state_to_rc(s)
        a = int(np.argmax(agent.Q[s]))
        ns, rw, done = env.step(a)
        r1, c1 = env.state_to_rc(ns)
        greedy_path.append(f"({r0},{c0}){action_symbols[a]}({r1},{c1})")
        if done:
            greedy_path.append("DONE")
            break
    logger.info(f"Greedy path from start: {' → '.join(greedy_path)}")

    # ------------------------------------------------------------------
    # 6. Learning curve (Part C.2)
    # ------------------------------------------------------------------
    plot_learning_curve(
        rewards,
        rolling_avg,
        window=config.convergence_window,
        plots_dir=plots_dir,
    )

    # ------------------------------------------------------------------
    # 7. Policy visualisation (Part D.1)
    # ------------------------------------------------------------------
    visualize_policy(env, agent, plots_dir=plots_dir)

    # ------------------------------------------------------------------
    # 8. Value function heatmap (Part D.2)
    # ------------------------------------------------------------------
    visualize_value_function(env, agent, plots_dir=plots_dir)

    logger.info("=" * 65)
    logger.info(f"All outputs saved under: {exp_dir}")
    logger.info("Experiment completed successfully.")
    logger.info("=" * 65)
