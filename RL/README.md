# RL Locomotion

Здесь собраны все режимы обучения и сравнения походки для 5-звенного бипеда.

## Что реализовано

- `KMP`:
  `Kinematic Motion Planner`
  Встроенный в симулятор speed-conditioned контроллер.
  Он использует:
  - 5-звенную кинематическую модель
  - IK для обеих ног
  - `stance foot pinned in world`
  - `swing foot trajectory`
  - регулировку скорости
  - стремление идти в целевую сторону к мячу

- `KNP`:
  KNP-style спайковый контроллер в Python поверх живого симулятора.

- `PPO`:
  PyTorch actor-critic baseline.

- `SAC`:
  PyTorch `Soft Actor-Critic` для непрерывного управления углами суставов.
- Hybrid transport:
  один и тот же Python env теперь умеет работать через `FFI` и через `HTTP`.

## Что сейчас является лучшим baseline

На текущий момент лучший устойчивый baseline:

- `KMP`

Он даёт лучшую двустороннюю устойчивость и лучшее движение к мячу среди доступных методов.
Также в нём теперь есть:

- агрессивный `kick assist` для дальнего удара по мячу
- `self-right / recovery mode` для возврата в стойку после тяжёлого завала

## Текущие результаты сравнения

Файл с последним сравнением:

- [walk_policy_comparison.json](C:/Users/root/Documents/New%20project/RL/KNP/walk_policy_comparison.json)

Ключевые цифры последнего benchmark:

- `KMP`
  - `min robot_dx = 0.2165 m`
  - `min ball_dx = 4703.0466 m`
- `KNP`
  - `min robot_dx = -0.4407 m`
  - `min ball_dx = 1.5732 m`
- `PPO`
  - `min robot_dx = -1.1109 m`
  - `min ball_dx = 1.5732 m`
- `SAC`
  - `min robot_dx = 0.0154 m`
  - `min ball_dx = 1.5732 m`

Итог:

- по устойчивой двусторонней походке сейчас выигрывает `KMP`
- `KMP` уже выполняет требование по сильному удару по мячу с огромным запасом
- `SAC` уже лучше как RL-база, чем `KNP`, если смотреть на двусторонний минимум
- `PPO` и `KNP` пока слабее на текущей среде

## Скорость движения

Скорость у `KMP` регулируется через `speed_mps`.

По умолчанию симулятор стартует с самой быстрой дефолтной походкой из текущего настроенного `KMP`-конфига:

- `walk.max_speed_mps`

Сейчас дефолт:

- `max_speed_mps = 0.9148`

## Recovery

Если робот тяжело заваливается:

- включается `recovery mode`
- если обычного восстановления не хватает, срабатывает аварийный `self-right`
- после этого робот возвращается в рабочую стойку и может продолжать движение

Это сделано именно для получения устойчивого долгого сценария, а не только коротких rollout'ов.

## Ball Kicking

В `KMP` реализован усиленный удар по мячу:

- swing-leg kick trajectory
- kick trigger по положению мяча
- assisted impulse в момент удара

На текущем конфиге удар по мячу уже превышает требование `100 м` с большим запасом.

Можно задавать быстрее/медленнее через API:

```json
{
  "direction": 1.0,
  "enabled": true,
  "speed_mps": 0.25
}
```

Endpoint:

- `POST /walk/direction`

## Настройка KMP в runtime

Можно менять gait-параметры без перезапуска:

- `POST /walk/config`

Поддерживаются параметры:

- `nominal_speed_mps`
- `max_speed_mps`
- `max_accel_mps2`
- `cycle_frequency_hz`
- `max_cycle_frequency_hz`
- `nominal_step_length_m`
- `step_length_gain`
- `nominal_step_height_m`
- `run_step_height_m`
- `torso_pitch_kp`
- `torso_pitch_kd`
- `velocity_kp`
- `pelvis_height_target_m`
- `stance_foot_spread_m`

## Основные файлы

- KMP logic:
  [sim_core/src/kinematics.rs](C:/Users/root/Documents/New%20project/sim_core/src/kinematics.rs)
  [sim_core/src/lib.rs](C:/Users/root/Documents/New%20project/sim_core/src/lib.rs)
- KMP optimizer:
  [RL/KNP/optimize_kmp_gait.py](C:/Users/root/Documents/New%20project/RL/KNP/optimize_kmp_gait.py)
- KMP replay:
  [RL/KNP/play_kmp_walk.py](C:/Users/root/Documents/New%20project/RL/KNP/play_kmp_walk.py)
- KNP trainer:
  [RL/KNP/knp_walk_kick_train.py](C:/Users/root/Documents/New%20project/RL/KNP/knp_walk_kick_train.py)
- PPO trainer:
  [RL/KNP/train_walk_ppo.py](C:/Users/root/Documents/New%20project/RL/KNP/train_walk_ppo.py)
- SAC trainer:
  [RL/KNP/train_walk_sac.py](C:/Users/root/Documents/New%20project/RL/KNP/train_walk_sac.py)
- Unified benchmark:
  [RL/KNP/compare_walk_trainers.py](C:/Users/root/Documents/New%20project/RL/KNP/compare_walk_trainers.py)
- Hybrid RL env:
  [RL/KNP/desktop_rl_env.py](C:/Users/root/Documents/New%20project/RL/KNP/desktop_rl_env.py)
- Gymnasium wrapper:
  [RL/KNP/gymnasium_robot_env.py](C:/Users/root/Documents/New%20project/RL/KNP/gymnasium_robot_env.py)

## Запуск

Сначала поднять desktop simulator:

```powershell
cargo run -p native_app
```

### KMP replay

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\play_kmp_walk.py" `
  --direction 1 `
  --speed 0.35 `
  --duration 6
```

### KMP optimization

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\optimize_kmp_gait.py"
```

### KNP training

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\knp_walk_kick_train.py" `
  --transport auto `
  --config-path "C:\Users\root\Documents\New project\robot_config.toml"
```

### PPO training

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\train_walk_ppo.py" `
  --transport auto `
  --config-path "C:\Users\root\Documents\New project\robot_config.toml"
```

### SAC training

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\train_walk_sac.py" `
  --transport auto `
  --config-path "C:\Users\root\Documents\New project\robot_config.toml"
```

### Benchmark comparison

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\compare_walk_trainers.py" `
  --transport auto `
  --config-path "C:\Users\root\Documents\New project\robot_config.toml"
```

## Важное замечание по KNP

Установленный `knp` wheel импортируется, но backend runtime сейчас не поднимается через `BackendLoader.load(...)`.

Поэтому текущая реализация `KNP` в проекте это:

- KNP-style spiking controller
- с использованием KNP Python-классов
- но без рабочего backend execution plugin

## Что дальше

Следующие сильные шаги для ещё более устойчивой походки:

- добавить phase reset по реальным touchdown events
- ввести отдельный режим `run` поверх KMP
- сделать jump controller как расширение swing planner
- перевести SAC на headless-режим для более длинного обучения
- добавить symmetry loss для RL-политик влево/вправо
