use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub physics: PhysicsConfig,
    pub robot: RobotConfig,
    pub servo: ServoConfig,
    #[serde(default)]
    pub walk: WalkConfig,
    #[serde(default)]
    pub rl: RlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub gravity_y: f32,
    pub dt: f32,
    pub ground_half_width: f32,
    pub ground_friction: f32,
    pub ground_restitution: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    pub torso: LinkConfig,
    pub thigh: LinkConfig,
    pub shin: LinkConfig,
    pub body_dynamics: BodyDynamicsConfig,
    pub suspend_clearance: f32,
    pub ball_spawn_offset_x_m: f32,
    pub initial_pose: InitialPoseConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkConfig {
    pub length: f32,
    pub width: f32,
    pub mass: f32,
    pub friction: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyDynamicsConfig {
    pub angular_damping: f32,
    pub linear_damping: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialPoseConfig {
    pub torso: BodyPoseConfig,
    pub left_thigh: BodyPoseConfig,
    pub left_shin: BodyPoseConfig,
    pub right_thigh: BodyPoseConfig,
    pub right_shin: BodyPoseConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BodyPoseConfig {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JointAnglesConfig {
    pub right_hip: f32,
    pub right_knee: f32,
    pub left_hip: f32,
    pub left_knee: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServoConfig {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub max_torque: f32,
    pub max_speed_deg_s: f32,
    pub integral_limit: f32,
    pub voltage_min_v: f32,
    pub voltage_max_v: f32,
    pub nominal_voltage_v: f32,
    pub stall_current_a: f32,
    pub no_load_current_a: f32,
    pub encoder_bits: u32,
    pub weight_kg: f32,
    pub gear_ratio: f32,
    pub zero_offsets: JointAnglesConfig,
    pub initial_targets: JointAnglesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkConfig {
    pub nominal_speed_mps: f32,
    pub max_speed_mps: f32,
    pub max_accel_mps2: f32,
    pub cycle_frequency_hz: f32,
    pub max_cycle_frequency_hz: f32,
    pub nominal_step_length_m: f32,
    pub step_length_gain: f32,
    pub nominal_step_height_m: f32,
    pub run_step_height_m: f32,
    pub stance_duty_factor: f32,
    pub torso_pitch_kp: f32,
    pub torso_pitch_kd: f32,
    pub hip_upright_gain: f32,
    pub hip_upright_damping: f32,
    pub torso_upright_limit_rad: f32,
    pub torso_forward_lean_per_speed: f32,
    pub torso_forward_lean_max_rad: f32,
    pub velocity_kp: f32,
    pub pelvis_height_target_m: f32,
    pub stance_foot_spread_m: f32,
    pub foot_separation_min_m: f32,
    pub kick_trigger_distance_m: f32,
    pub kick_forward_impulse_ns: f32,
    pub kick_upward_impulse_ns: f32,
    pub recovery_height_threshold_m: f32,
    pub recovery_angle_threshold_rad: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            nominal_speed_mps: 0.25145823,
            max_speed_mps: 0.25145823,
            max_accel_mps2: 0.2,
            cycle_frequency_hz: 0.50172114,
            max_cycle_frequency_hz: 3.648091,
            nominal_step_length_m: 0.065191515,
            step_length_gain: 0.034639854,
            nominal_step_height_m: 0.01,
            run_step_height_m: 0.056611225,
            stance_duty_factor: 0.61050344,
            torso_pitch_kp: 1.1255612,
            torso_pitch_kd: 0.3615783,
            hip_upright_gain: 0.787138,
            hip_upright_damping: 0.30061188,
            torso_upright_limit_rad: 0.04150689,
            torso_forward_lean_per_speed: 0.12861174,
            torso_forward_lean_max_rad: 0.04150689,
            velocity_kp: 0.23928972,
            pelvis_height_target_m: 0.87267643,
            stance_foot_spread_m: 0.10186975,
            foot_separation_min_m: 0.1452655,
            kick_trigger_distance_m: 1.2,
            kick_forward_impulse_ns: 120.0,
            kick_upward_impulse_ns: 12.0,
            recovery_height_threshold_m: 0.58,
            recovery_angle_threshold_rad: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    pub control_substeps: u32,
    pub episode_timeout_s: f32,
    pub torso_min_height: f32,
    pub torso_max_tilt_rad: f32,
    pub action_limit_deg: f32,
    pub reward_forward_weight: f32,
    pub reward_alive_bonus: f32,
    pub reward_upright_weight: f32,
    pub reward_height_weight: f32,
    pub reward_contact_weight: f32,
    pub reward_ball_forward_weight: f32,
    pub penalty_torque_weight: f32,
    pub penalty_action_delta_weight: f32,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            control_substeps: 4,
            episode_timeout_s: 20.0,
            torso_min_height: 0.30,
            torso_max_tilt_rad: 1.25,
            action_limit_deg: 180.0,
            reward_forward_weight: 3.0,
            reward_alive_bonus: 0.05,
            reward_upright_weight: 0.45,
            reward_height_weight: 0.25,
            reward_contact_weight: 0.10,
            reward_ball_forward_weight: 0.20,
            penalty_torque_weight: 0.0035,
            penalty_action_delta_weight: 0.0040,
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsConfig {
                gravity_y: -9.81,
                dt: 1.0 / 120.0,
                ground_half_width: 5_000.0,
                ground_friction: 1.35,
                ground_restitution: 0.0,
            },
            robot: RobotConfig {
                torso: LinkConfig {
                    length: 0.68,
                    width: 0.09,
                    mass: 0.318_240_02,
                    friction: 1.0,
                },
                thigh: LinkConfig {
                    length: 0.46,
                    width: 0.09,
                    mass: 0.140_760_02,
                    friction: 1.0,
                },
                shin: LinkConfig {
                    length: 0.50,
                    width: 0.09,
                    mass: 0.126,
                    friction: 1.2,
                },
                body_dynamics: BodyDynamicsConfig {
                    angular_damping: 0.85,
                    linear_damping: 0.08,
                },
                suspend_clearance: 0.8,
                ball_spawn_offset_x_m: 12.0,
                initial_pose: InitialPoseConfig {
                    torso: BodyPoseConfig {
                        x: 0.0,
                        y: 1.205,
                        angle: 0.0,
                    },
                    left_thigh: BodyPoseConfig {
                        x: 0.014,
                        y: 0.62,
                        angle: 0.05,
                    },
                    left_shin: BodyPoseConfig {
                        x: -0.11,
                        y: 0.19,
                        angle: -0.82,
                    },
                    right_thigh: BodyPoseConfig {
                        x: -0.014,
                        y: 0.62,
                        angle: -0.05,
                    },
                    right_shin: BodyPoseConfig {
                        x: 0.11,
                        y: 0.19,
                        angle: 0.82,
                    },
                },
            },
            servo: ServoConfig {
                kp: 20.0,
                ki: 0.0,
                kd: 0.0,
                max_torque: 2.55,
                max_speed_deg_s: 439.0,
                integral_limit: 10.0,
                voltage_min_v: 6.0,
                voltage_max_v: 15.0,
                nominal_voltage_v: 12.0,
                stall_current_a: 3.0,
                no_load_current_a: 0.2,
                encoder_bits: 14,
                weight_kg: 0.046,
                gear_ratio: 330.0,
                zero_offsets: JointAnglesConfig {
                    right_hip: -0.05,
                    right_knee: 0.87,
                    left_hip: 0.05,
                    left_knee: -0.87,
                },
                initial_targets: JointAnglesConfig {
                    right_hip: -0.05,
                    right_knee: 0.87,
                    left_hip: 0.05,
                    left_knee: -0.87,
                },
            },
            walk: WalkConfig::default(),
            rl: RlConfig::default(),
        }
    }
}

impl Default for JointAnglesConfig {
    fn default() -> Self {
        SimulationConfig::default().servo.zero_offsets
    }
}

impl SimulationConfig {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|err| format!("failed to read config '{}': {err}", path.display()))?;
        toml::from_str(&text)
            .map_err(|err| format!("failed to parse config '{}': {err}", path.display()))
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let text = toml::to_string_pretty(self)
            .map_err(|err| format!("failed to serialize config '{}': {err}", path.display()))?;
        fs::write(path, text).map_err(|err| format!("failed to write config '{}': {err}", path.display()))
    }
}
