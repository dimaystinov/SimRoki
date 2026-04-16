
# Полное руководство: Обучение двуногих походок методом Periodic Reward Composition
## На основе статьи ICRA 2021: Siekmann et al. "Sim-to-Real Learning of All Common Bipedal Gaits"

---

## 1. Аннотация

Статья представляет фреймворк спецификации наград на основе периодической композиции простых вероятностных функций стоимости от базовых сил и скоростей. Этот подход позволяет обучить все распространенные двуногие походки (стояние, ходьбу, бег, прыжки, скакание) и перенести их на реального робота Cassie.

---

## 2. Математическая формализация

### 2.1 Периодическая композиция наград

Ключевая инновация - использование вероятностных периодических функций:

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

class PeriodicRewardComposer:
    """
    Композитор периодических наград для обучения двуногих походок.
    Реализация на основе статьи Siekmann et al. (ICRA 2021).
    """

    def __init__(self, 
                 cycle_time: float = 0.6,
                 phase_offset_left: float = 0.0,
                 phase_offset_right: float = 0.5,
                 swing_ratio: float = 0.4):
        self.cycle_time = cycle_time
        self.phase_offset_left = phase_offset_left
        self.phase_offset_right = phase_offset_right
        self.swing_ratio = swing_ratio

        # Параметры для всех походок
        self.gait_params = self._init_gait_parameters()

    def _init_gait_parameters(self) -> Dict[str, Dict]:
        """Инициализация параметров для всех распространенных походок."""
        return {
            'standing': {
                'cycle_time': 1.0,
                'phase_offset_left': 0.0,
                'phase_offset_right': 0.0,
                'swing_ratio': 0.0,
                'velocity_target': 0.0,
                'height_target': 0.9
            },
            'walking': {
                'cycle_time': 0.6,
                'phase_offset_left': 0.0,
                'phase_offset_right': 0.5,
                'swing_ratio': 0.4,
                'velocity_target': 1.0,
                'height_target': 0.9
            },
            'running': {
                'cycle_time': 0.4,
                'phase_offset_left': 0.0,
                'phase_offset_right': 0.5,
                'swing_ratio': 0.5,
                'velocity_target': 3.0,
                'height_target': 0.95
            },
            'hopping': {
                'cycle_time': 0.5,
                'phase_offset_left': 0.0,
                'phase_offset_right': 0.0,
                'swing_ratio': 0.5,
                'velocity_target': 0.0,
                'height_target': 1.0
            },
            'skipping': {
                'cycle_time': 0.6,
                'phase_offset_left': 0.0,
                'phase_offset_right': 0.25,
                'swing_ratio': 0.3,
                'velocity_target': 2.0,
                'height_target': 1.0
            }
        }

    def get_phase(self, time: float, phase_offset: float) -> float:
        """Вычисление текущей фазы в цикле [0, 1]."""
        return ((time / self.cycle_time) + phase_offset) % 1.0

    def probabilistic_contact_cost(self, 
                                  phase: float, 
                                  contact_force: float) -> float:
        """
        Вероятностная функция стоимости контакта ноги с землей.
        Ключевая инновация статьи - использование вероятностного подхода
        для определения, должна ли нога находиться в контакте в данной фазе.
        """
        # Желаемая вероятность контакта
        if phase < self.swing_ratio:
            desired_contact_prob = 0.0
        else:
            desired_contact_prob = 1.0

        # Плавный переход через сигмоид
        phase_in_stance = (phase - self.swing_ratio) / (1.0 - self.swing_ratio + 1e-8)
        desired_contact_prob = torch.sigmoid(torch.tensor((phase_in_stance - 0.5) * 10.0))

        # Нормализация силы контакта
        normalized_force = contact_force / 100.0

        # Вероятностная стоимость
        cost = (desired_contact_prob.item() - normalized_force) ** 2

        return cost
```

### 2.2 Полная функция награды

```python
class CassieRewardFunction:
    """Полная функция награды для робота Cassie."""

    def __init__(self, reward_weights: Dict[str, float] = None):
        self.composer = PeriodicRewardComposer()

        # Веса из статьи
        self.weights = reward_weights or {
            'velocity_tracking': 1.0,
            'orientation': 0.5,
            'periodic_contact_left': 1.0,
            'periodic_contact_right': 1.0,
            'smooth_action': 0.1,
            'energy_efficiency': 0.01,
            'alive_bonus': 1.0
        }

    def compute_reward(self, state: Dict, action: np.ndarray, 
                      time: float, gait_type: str = 'walking') -> Tuple[float, Dict]:
        """Вычисление полной награды."""

        # Установка параметров походки
        params = self.composer.gait_params[gait_type]
        self.composer.cycle_time = params['cycle_time']
        self.composer.swing_ratio = params['swing_ratio']

        # Вычисление фаз
        phase_left = self.composer.get_phase(time, params['phase_offset_left'])
        phase_right = self.composer.get_phase(time, params['phase_offset_right'])

        # 1. Награда за отслеживание скорости
        vel_error = np.linalg.norm(
            state['pelvis_velocity'] - np.array([params['velocity_target'], 0.0, 0.0])
        )
        vel_reward = -self.weights['velocity_tracking'] * vel_error ** 2

        # 2. Награда за ориентацию
        orient_reward = -self.weights['orientation'] * self._orientation_cost(
            state['pelvis_orientation']
        )

        # 3. Периодические награды за контакт ног (КЛЮЧЕВАЯ ИННОВАЦИЯ)
        periodic_left = -self.weights['periodic_contact_left'] *                        self.composer.probabilistic_contact_cost(
                           phase_left, state['left_foot_contact_force']
                       )

        periodic_right = -self.weights['periodic_contact_right'] *                         self.composer.probabilistic_contact_cost(
                            phase_right, state['right_foot_contact_force']
                        )

        # 4. Награда за плавность
        smooth_reward = 0.0
        if 'previous_action' in state:
            smooth_reward = -self.weights['smooth_action'] *                            np.linalg.norm(action - state['previous_action'])

        # 5. Награда за энергоэффективность
        torque_penalty = -self.weights['energy_efficiency'] *                         np.sum(np.square(state['joint_torques']))

        # 6. Бонус за выживание
        alive_reward = self.weights['alive_bonus']

        total_reward = (vel_reward + orient_reward + 
                       periodic_left + periodic_right + 
                       smooth_reward + torque_penalty + alive_reward)

        reward_info = {
            'velocity': vel_reward,
            'orientation': orient_reward,
            'periodic_left': periodic_left,
            'periodic_right': periodic_right,
            'smoothness': smooth_reward,
            'energy': torque_penalty,
            'alive': alive_reward,
            'total': total_reward,
            'phase_left': phase_left,
            'phase_right': phase_right
        }

        return total_reward, reward_info

    def _orientation_cost(self, quat: np.ndarray) -> float:
        """Стоимость отклонения от вертикальной ориентации."""
        target = np.array([1.0, 0.0, 0.0, 0.0])
        dot_product = np.abs(np.dot(quat, target))
        return 1.0 - dot_product ** 2
```

---

## 3. Среда симуляции

### 3.1 Установка зависимостей

```bash
# 1. Установка MuJoCo 2.1.0
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz

# 2. Настройка переменных окружения
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_KEY_PATH=~/.mujoco/mjkey.txt

# 3. Клонирование репозиториев Cassie
git clone https://github.com/osudrl/cassie-mujoco-sim.git
git clone https://github.com/siekmanj/cassie.git

# 4. Сборка cassie-mujoco-sim
cd cassie-mujoco-sim
make
make test

# 5. Установка Python-зависимостей
pip install torch numpy gymnasium mujoco
```

### 3.2 Python-среда Cassie

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cassiemujoco

class CassieEnv(gym.Env):
    """Среда Gym для робота Cassie на основе MuJoCo."""

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, gait_type='walking', simrate=50, 
                 dynamics_randomization=True):
        super().__init__()

        self.gait_type = gait_type
        self.simrate = simrate
        self.dynamics_randomization = dynamics_randomization

        # Инициализация симулятора
        self.sim = cassiemujoco.CassieSim()
        self.vis = None

        # Пространство действий: 10 моторов
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,))

        # Пространство наблюдений: 37 размерностей
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,))

        # Функция награды
        self.reward_fn = CassieRewardFunction()

        # Время
        self.time = 0.0
        self.timestep = 0.0005  # 2000 Hz

        # Параметры рандомизации
        self.dynamics_ranges = {
            'joint_damping': (0.5, 1.5),
            'joint_mass': (0.8, 1.2),
            'ground_friction': (0.5, 1.5),
        }

    def _get_observation(self) -> np.ndarray:
        """Получение наблюдения от робота."""
        qpos = self.sim.qpos()
        qvel = self.sim.qvel()

        # Ориентация таза
        pelvis_quat = qpos[3:7]
        pelvis_rot_vel = qvel[3:6]

        # Суставы
        joint_pos = qpos[7:21]
        joint_vel = qvel[6:20]

        # Фаза
        phase = np.array([
            np.sin(2 * np.pi * self.time / self.reward_fn.composer.cycle_time),
            np.cos(2 * np.pi * self.time / self.reward_fn.composer.cycle_time)
        ])

        obs = np.concatenate([pelvis_quat, pelvis_rot_vel, 
                             joint_pos, joint_vel, phase])

        return obs.astype(np.float32)

    def _apply_dynamics_randomization(self):
        """Применение рандомизации динамики."""
        if not self.dynamics_randomization:
            return

        damping = np.random.uniform(*self.dynamics_ranges['joint_damping'])
        mass = np.random.uniform(*self.dynamics_ranges['joint_mass'])
        friction = np.random.uniform(*self.dynamics_ranges['ground_friction'])

        self.sim.set_joint_damping(damping)
        self.sim.set_body_mass(mass)
        self.sim.set_ground_friction(friction)

    def reset(self, seed=None, options=None):
        """Сброс среды."""
        super().reset(seed=seed)

        self.sim.reset()
        self.time = 0.0

        if self.dynamics_randomization:
            self._apply_dynamics_randomization()

        # Случайные возмущения
        noise = np.random.randn(self.sim.nq) * 0.01
        self.sim.set_qpos(self.sim.qpos() + noise)

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Шаг симуляции."""
        # Денормализация действий
        motor_targets = self._denormalize_actions(action)

        # Выполнение шагов симуляции
        for _ in range(self.simrate):
            self.sim.step_pd(motor_targets)
            self.time += self.timestep

        # Получение нового состояния
        obs = self._get_observation()

        # Формирование состояния для награды
        state = {
            'pelvis_velocity': self.sim.pelvis_vel(),
            'pelvis_orientation': self.sim.pelvis_quat(),
            'left_foot_contact_force': self.sim.left_foot_force(),
            'right_foot_contact_force': self.sim.right_foot_force(),
            'joint_torques': self.sim.joint_torques(),
            'previous_action': getattr(self, 'prev_action', None)
        }

        # Вычисление награды
        reward, reward_info = self.reward_fn.compute_reward(
            state, action, self.time, self.gait_type
        )

        # Проверка завершения
        terminated = self._check_termination()
        truncated = self.time > 10.0

        info = {'reward_info': reward_info, 'time': self.time}

        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def _denormalize_actions(self, normalized_actions):
        """Преобразование [-1, 1] в реальные углы."""
        ranges = [
            (-0.2618, 0.2618),   # hip roll
            (-0.4014, 0.4014),   # hip yaw
            (-1.0472, 1.0472),   # hip pitch
            (0.0, 1.8326),       # knee
            (-1.0472, 1.0472),   # foot
        ]

        targets = np.zeros(10)
        for leg in range(2):
            for joint in range(5):
                idx = leg * 5 + joint
                low, high = ranges[joint]
                targets[idx] = low + (normalized_actions[idx] + 1.0) * 0.5 * (high - low)

        return targets

    def _check_termination(self):
        """Проверка условий падения."""
        if self.sim.pelvis_height() < 0.5:
            return True

        pelvis_quat = self.sim.pelvis_quat()
        w, x, y, z = pelvis_quat
        rotated_up_z = 2 * (w * y - x * z)
        if rotated_up_z < 0.5:
            return True

        return False

    def render(self):
        if self.vis is None:
            self.vis = cassiemujoco.CassieVis(self.sim)
        return self.vis.draw()

    def close(self):
        if self.vis is not None:
            self.vis.close()
```

---

## 4. Алгоритм обучения (PPO)

### 4.1 Архитектура политики с LSTM

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticPolicy(nn.Module):
    """Политика Actor-Critic с LSTM для Cassie."""

    def __init__(self, obs_dim=37, action_dim=10, 
                 hidden_dim=256, lstm_hidden=128):
        super().__init__()

        self.lstm = nn.LSTM(obs_dim, lstm_hidden, batch_first=True)

        # Actor (политика)
        self.actor_mean = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Логарифм std (обучаемый)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (ценность)
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, hidden=None):
        """Прямой проход."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        lstm_out, new_hidden = self.lstm(obs, hidden)
        features = lstm_out[:, -1, :]

        action_mean = self.actor_mean(features)
        value = self.critic(features)
        std = torch.exp(self.log_std)

        return action_mean, value, new_hidden, std

    def get_action(self, obs, hidden=None, deterministic=False):
        """Получение действия."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mean, value, new_hidden, std = self.forward(obs_tensor, hidden)

            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()

            return (action.squeeze(0).numpy(), value.item(), 
                   new_hidden)

    def evaluate_actions(self, obs, actions, hidden=None):
        """Оценка действий для обучения."""
        mean, values, _, std = self.forward(obs, hidden)

        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy
```

### 4.2 PPO Тренер

```python
class PPOTrainer:
    """Proximal Policy Optimization для Cassie."""

    def __init__(self, env, policy, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_eps=0.2, device='cpu'):
        self.env = env
        self.policy = policy.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        # Буферы
        self.reset_buffers()

    def reset_buffers(self):
        """Очистка буферов."""
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []

    def collect_trajectories(self, num_steps, hidden_init=None):
        """Сбор траекторий."""
        obs, _ = self.env.reset()
        hidden = hidden_init

        for step in range(num_steps):
            action, value, new_hidden = self.policy.get_action(obs, hidden)

            self.obs_buffer.append(obs)
            self.action_buffer.append(action)
            self.value_buffer.append(value)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.reward_buffer.append(reward)
            self.done_buffer.append(done)

            obs = next_obs
            hidden = new_hidden

            if done:
                obs, _ = self.env.reset()
                hidden = None

    def compute_gae(self, next_value):
        """Generalized Advantage Estimation."""
        rewards = np.array(self.reward_buffer)
        values = np.array(self.value_buffer + [next_value])
        dones = np.array(self.done_buffer)

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_val = 0
                last_gae = 0
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update_policy(self, next_obs):
        """Обновление политики."""
        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)

        with torch.no_grad():
            _, next_value, _, _ = self.policy.forward(
                torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            )

        advantages, returns = self.compute_gae(next_value.item())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Получение старых лог-вероятностей
        with torch.no_grad():
            old_log_probs, _, _ = self.policy.evaluate_actions(obs, actions)

        # PPO эпохи
        for epoch in range(10):
            log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)

            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * ((values - returns) ** 2).mean()

            # Итоговый loss
            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.reset_buffers()

    def train(self, total_timesteps=10_000_000, steps_per_update=4096):
        """Основной цикл обучения."""
        timesteps = 0

        while timesteps < total_timesteps:
            self.collect_trajectories(steps_per_update)

            last_obs = self.obs_buffer[-1] if self.obs_buffer else self.env.reset()[0]
            self.update_policy(last_obs)

            timesteps += steps_per_update

            if timesteps % 100_000 == 0:
                print(f"Timesteps: {timesteps}/{total_timesteps}")
                self.save_checkpoint(f"checkpoint_{timesteps}.pt")

    def save_checkpoint(self, path):
        """Сохранение модели."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
```

---

## 5. Обучение универсальной политики (Multi-Gait)

```python
class MultiGaitTrainer:
    """Обучение политики для всех походок сразу."""

    def __init__(self, base_trainer):
        self.trainer = base_trainer
        self.gait_params = PeriodicRewardComposer().gait_params

    def sample_gait_command(self):
        """Сэмплирование случайной походки."""
        gait_names = list(self.gait_params.keys())
        gait_name = np.random.choice(gait_names)
        return gait_name, self.gait_params[gait_name]

    def train_multi_gait(self, total_timesteps=20_000_000):
        """Обучение на всех походках."""
        timesteps = 0

        while timesteps < total_timesteps:
            # Случайный выбор походки
            gait_name, params = self.sample_gait_command()

            # Установка параметров
            self.trainer.env.reward_fn.composer.cycle_time = params['cycle_time']
            self.trainer.env.reward_fn.composer.swing_ratio = params['swing_ratio']
            self.trainer.env.gait_type = gait_name

            # Сбор и обновление
            self.trainer.collect_trajectories(4096)
            last_obs = self.trainer.obs_buffer[-1]
            self.trainer.update_policy(last_obs)

            timesteps += 4096

            if timesteps % 100000 == 0:
                print(f"Timesteps: {timesteps}, Current gait: {gait_name}")
```

---

## 6. Sim-to-Real Transfer

### 6.1 Рандомизация динамики

Критично для переноса на реального робота:

```python
class DynamicsRandomizer:
    """Рандомизация параметров динамики."""

    RANGES = {
        'joint_damping': (0.5, 2.0),
        'joint_mass': (0.8, 1.2),
        'ground_friction': (0.5, 1.5),
        'motor_strength': (0.8, 1.2),
        'sensor_noise': (0.0, 0.02),
    }

    def randomize(self, sim):
        """Применение случайных параметров."""
        for param, (low, high) in self.RANGES.items():
            value = np.random.uniform(low, high)
            if 'damping' in param:
                sim.set_joint_damping(value)
            elif 'mass' in param:
                sim.set_body_mass(value)
            elif 'friction' in param:
                sim.set_ground_friction(value)
```

### 6.2 Развертывание на реальном роботе

```python
class CassieRealDeploy:
    """Развертывание на реальном Cassie."""

    def __init__(self, policy_path, robot_ip='192.168.1.1'):
        # Загрузка политики
        self.policy = ActorCriticPolicy()
        checkpoint = torch.load(policy_path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        # Соединение с роботом
        from cassie_udp import CassieUDP
        self.robot = CassieUDP(robot_ip)

        self.hidden = None
        self.control_freq = 40  # Hz

    def get_observation(self):
        """Получение данных от робота."""
        packet = self.robot.recv()

        # Ориентация
        quat = np.array([
            packet.pelvis.orientation.w,
            packet.pelvis.orientation.x,
            packet.pelvis.orientation.y,
            packet.pelvis.orientation.z
        ])

        # Угловая скорость
        rot_vel = np.array([
            packet.pelvis.rotationalVelocity.x,
            packet.pelvis.rotationalVelocity.y,
            packet.pelvis.rotationalVelocity.z
        ])

        # Суставы
        motor_pos = np.array(packet.motor.position)
        motor_vel = np.array(packet.motor.velocity)
        joint_pos = np.array(packet.joint.position)
        joint_vel = np.array(packet.joint.velocity)

        all_pos = np.concatenate([motor_pos, joint_pos])
        all_vel = np.concatenate([motor_vel, joint_vel])

        # Фаза (по времени)
        phase = np.array([0.0, 1.0])

        return np.concatenate([quat, rot_vel, all_pos, all_vel, phase])

    def run(self):
        """Основной цикл."""
        import time

        print("Starting real robot control...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                start = time.time()

                # Получение наблюдения
                obs = self.get_observation()

                # Получение действия
                with torch.no_grad():
                    action, _, self.hidden = self.policy.get_action(
                        obs, self.hidden, deterministic=True
                    )

                # Отправка на робота
                targets = self.denormalize(action)
                self.send_command(targets)

                # Поддержание частоты
                elapsed = time.time() - start
                if elapsed < 1/self.control_freq:
                    time.sleep(1/self.control_freq - elapsed)

        except KeyboardInterrupt:
            print("Stopping...")
            self.send_zero_command()

    def denormalize(self, action):
        """Преобразование в углы моторов."""
        ranges = [
            (-0.2618, 0.2618), (-0.4014, 0.4014), (-1.0472, 1.0472),
            (0.0, 1.8326), (-1.0472, 1.0472),
        ]

        targets = np.zeros(10)
        for i in range(10):
            joint = i % 5
            low, high = ranges[joint]
            targets[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        return targets

    def send_command(self, targets):
        """Отправка команды на Cassie."""
        pass

    def send_zero_command(self):
        """Безопасная остановка."""
        safe = np.array([0.0, 0.0, 0.0, 0.5, 0.0] * 2)
        self.send_command(safe)
```

---

## 7. Запуск обучения

```python
#!/usr/bin/env python3
"""Скрипт запуска обучения."""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gait', default='walking',
                       choices=['standing', 'walking', 'running', 
                               'hopping', 'skipping', 'multi'])
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()

    # Настройка
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # Создание среды
    env = CassieEnv(
        gait_type=args.gait,
        dynamics_randomization=True  # Критично для sim-to-real!
    )

    # Создание политики
    policy = ActorCriticPolicy()

    # Тренер
    trainer = PPOTrainer(env, policy, device=device)

    # Обучение
    if args.gait == 'multi':
        multi_trainer = MultiGaitTrainer(trainer)
        multi_trainer.train_multi_gait(args.timesteps)
    else:
        trainer.train(args.timesteps)

if __name__ == '__main__':
    main()
```

---

## 8. Ключевые выводы из статьи

1. **Периодическая композиция наград** позволяет интуитивно задавать различные походки через параметры фазы и времени цикла.

2. **Вероятностный подход** к контакту ног (вместо жестких ограничений) делает обучение более робастным.

3. **LSTM политика** необходима для запоминания фазы цикла и истории движения.

4. **Рандомизация динамики** критична для sim-to-real transfer.

5. **Универсальная политика** может переключаться между походками путем изменения параметров цикла.

---

## Ссылки

- **Оригинальная статья**: https://arxiv.org/abs/2011.01387
- **Видео**: https://www.youtube.com/watch?v=HbdA...
- **GitHub Cassie**: https://github.com/siekmanj/cassie
- **GitHub cassie-mujoco-sim**: https://github.com/osudrl/cassie-mujoco-sim
