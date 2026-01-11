"""
Versión auto-contenida de agent_trainer.py
Incluye:
 - Clase base Agent
 - Implementación QLearningAgent
 - Clase base Environment
 - Implementación MouseMaze (ejemplo simple)
 - AgentTrainer con el mismo flujo de entrenamiento que el archivo original

Este archivo permite ejecutar el entrenamiento sin depender de imports externos.
"""
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ----------------------------- AGENT (base) -----------------------------
class Agent(ABC):
    """Clase base para agentes de RL (ε-greedy + decay)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    @abstractmethod
    def choose_action(self, state: int, explore: bool = True) -> Tuple[int, str]:
        """Selecciona y devuelve (action, method).
        method = 'explore'|'exploit'
        """

    @abstractmethod
    def update_q_value(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> float:
        """Actualiza los valores internos del agente y devuelve el TD error."""

    def decay_epsilon(self):
        """Reduce epsilon para disminuir exploración."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ------------------------- Q-LEARNING AGENT ------------------------------
class QLearningAgent(Agent):
    """Agente que aprende usando Q-Learning (tabla discreta)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )

        # Inicializar Q-table con ceros
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state: int, explore: bool = True) -> Tuple[int, str]:
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            method = "explore"
        else:
            action = int(np.argmax(self.q_table[state]))
            method = "exploit"
        return action, method

    def update_q_value(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> float:
        current_q = self.q_table[state, action]
        max_next_q = 0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.lr * td_error
        return td_error


# --------------------------- ENVIRONMENT (base) --------------------------
class Environment(ABC):
    """Clase base para ambientes discretos simples."""

    def __init__(self, n_states: int, n_actions: int, max_steps: int = 100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.current_state = 0
        self.steps = 0
        self.action_names = {i: str(i) for i in range(n_actions)}

    def reset(self) -> int:
        self.current_state = 0
        self.steps = 0
        return self.current_state

    @abstractmethod
    def get_next_state(self, state: int, action: int) -> int:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        pass

    @abstractmethod
    def render(self, state: int) -> str:
        pass


# --------------------------- MOUSE MAZE (ejemplo) ------------------------
class MouseMaze(Environment):
    """Ejemplo: laberinto 3x3 con queso, veneno y pequeños quesos."""

    def __init__(self):
        self.grid_size = 3
        super().__init__(n_states=9, n_actions=4, max_steps=5)
        self.start_state = 0
        self.big_cheese_state = 8
        self.poison_state = 4
        self.small_cheese_states = [1, 2, 3, 5, 6, 7]
        self.rewards = {
            "big_cheese": 10,
            "small_cheese": 1,
            "poison": -10,
            "empty": 0,
            "timeout": 0,
        }
        self.current_state = self.start_state
        self.steps = 0
        self.action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def reset(self) -> int:
        self.current_state = self.start_state
        self.steps = 0
        return self.current_state

    def get_next_state(self, state: int, action: int) -> int:
        row, col = state // 3, state % 3
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(2, col + 1)
        elif action == 2:
            row = min(2, row + 1)
        elif action == 3:
            col = max(0, col - 1)
        return row * 3 + col

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        self.steps += 1
        next_state = self.get_next_state(self.current_state, action)

        if next_state == self.poison_state:
            reward = self.rewards["poison"]
            done = True
            info = {"reason": "poison"}
        elif next_state == self.big_cheese_state:
            reward = self.rewards["big_cheese"]
            done = True
            info = {"reason": "big_cheese"}
        elif next_state in self.small_cheese_states:
            reward = self.rewards["small_cheese"]
            done = False
            info = {"reason": "small_cheese"}
        else:
            reward = self.rewards["empty"]
            done = False
            info = {"reason": "empty"}

        if self.steps >= self.max_steps and not done:
            done = True
            reward = self.rewards["timeout"]
            info = {"reason": "timeout"}

        self.current_state = next_state
        return next_state, reward, done, info

    def render(self, state: int) -> str:
        size = self.grid_size
        grid = [["·" for _ in range(size)] for _ in range(size)]
        r, c = self.big_cheese_state // size, self.big_cheese_state % size
        grid[r][c] = "🧀"
        r, c = self.poison_state // size, self.poison_state % size
        grid[r][c] = "☠️"
        for s in self.small_cheese_states:
            r, c = s // size, s % size
            grid[r][c] = "·"
        r, c = state // size, state % size
        grid[r][c] = "🐭"
        return "\n".join([" ".join(row) for row in grid])


# --------------------------- AGENT TRAINER ------------------------------
class AgentTrainer:

    @staticmethod
    def train_agent(
        env: Environment,
        agent: Agent,
        n_episodes: int = 1000,
        verbose_every: int = 100,
        show_first_episodes: int = 5,
    ) -> Tuple[List[float], List[int], List[str]]:
        """Entrena al agente usando Q-Learning (flujo idéntico al original)."""
        print("\n" + "=" * 80)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 80)
        print(f"Episodios totales: {n_episodes}")
        print(f"Mostrando detalle de los primeros {show_first_episodes} episodios")
        print("=" * 80 + "\n")

        episode_rewards: List[float] = []
        episode_steps: List[int] = []
        episode_outcomes: List[str] = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            step = 0

            verbose = (episode < show_first_episodes) or (episode % verbose_every == 0)

            if verbose:
                print(f"\n{'='*80}")
                print(f"EPISODIO {episode + 1}/{n_episodes} | Epsilon: {getattr(agent, 'epsilon', 0):.4f}")
                print(f"{'='*80}")

            while not done:
                step += 1

                # Seleccionar acción
                action, method = agent.choose_action(state, explore=True)

                if verbose:
                    print(f"\n--- PASO {step} ---")
                    print(f"Estado actual: S{state}")
                    try:
                        q_vals = agent.q_table[state]
                        print(f"Q-values: {q_vals.round(3)}")
                    except Exception:
                        pass
                    print(f"Acción: {env.action_names[action]} (método: {method})")

                # Ejecutar acción
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                if verbose:
                    print(f"Siguiente estado: S{next_state}")
                    print(f"Recompensa: {reward} ({info.get('reason')})")
                    print(f"Terminado: {done}")

                # Actualizar Q-value
                td_error = agent.update_q_value(state, action, reward, next_state, done)

                if verbose:
                    print(f"\n--- ACTUALIZACIÓN Q-LEARNING ---")
                    try:
                        before = agent.q_table[state, action] - agent.lr * td_error
                        print(f"Q(S{state}, {env.action_names[action]}) antes: {before:.3f}")
                    except Exception:
                        pass
                    if not done:
                        try:
                            print(f"max Q(S{next_state}) = {np.max(agent.q_table[next_state]):.3f}")
                        except Exception:
                            pass
                    td_target = reward + agent.gamma * (np.max(agent.q_table[next_state]) if not done else 0)
                    print(f"TD Target = {reward} + {agent.gamma} * {np.max(agent.q_table[next_state]) if not done else 0:.3f} = {td_target:.3f}")
                    print(f"TD Error = {td_error:.3f}")
                    try:
                        print(f"Q(S{state}, {env.action_names[action]}) después: {agent.q_table[state, action]:.3f}")
                    except Exception:
                        pass

                state = next_state

            # Reducir epsilon
            agent.decay_epsilon()

            # Guardar métricas
            episode_rewards.append(total_reward)
            episode_steps.append(step)
            episode_outcomes.append(info.get("reason"))

            if verbose:
                print(f"\n{'='*80}")
                print(f"FIN EPISODIO {episode + 1}")
                print(f"{'='*80}")
                print(f"Recompensa total: {total_reward}")
                print(f"Pasos: {step}")
                print(f"Resultado: {info.get('reason')}")

            # Mostrar progreso cada N episodios
            if episode > 0 and episode % verbose_every == 0:
                recent_rewards = episode_rewards[-verbose_every:]
                print(f"\n📊 PROGRESO (Últimos {verbose_every} episodios)")
                print(f"   Recompensa promedio: {np.mean(recent_rewards):.2f}")
                print(f"   Recompensa máxima: {np.max(recent_rewards):.2f}")
                print(f"   Epsilon actual: {getattr(agent, 'epsilon', 0):.4f}")

        print("\n" + "=" * 80)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        print(f"\n📈 ESTADÍSTICAS FINALES:")
        print(f"   Recompensa promedio: {np.mean(episode_rewards):.2f}")
        print(f"   Recompensa máxima: {np.max(episode_rewards):.2f}")
        print(f"   Recompensa mínima: {np.min(episode_rewards):.2f}")
        print(f"   Pasos promedio: {np.mean(episode_steps):.2f}")

        outcomes_count = pd.Series(episode_outcomes).value_counts()
        print(f"\n📋 DISTRIBUCIÓN DE RESULTADOS:")
        for outcome, count in outcomes_count.items():
            print(f"   {outcome}: {count} ({count/ n_episodes * 100:.1f}%)")

        return episode_rewards, episode_steps, episode_outcomes


# --------------------------- EJEMPLO DE USO -----------------------------
if __name__ == "__main__":
    env = MouseMaze()
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    # Entrenar por una cantidad reducida de episodios para demostración
    rewards, steps, outcomes = AgentTrainer.train_agent(
        env, agent, n_episodes=200, verbose_every=50, show_first_episodes=2
    )

    print("\nResumen rápido:")
    print(f"  Episodios: {len(rewards)}")
    print(f"  Recompensa media: {np.mean(rewards):.2f}")
    print(f"  Recompensa máxima: {np.max(rewards):.2f}")
    print(f"  Resultados frecuentes: {pd.Series(outcomes).value_counts().head()}")
