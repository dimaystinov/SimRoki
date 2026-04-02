use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub physics: PhysicsConfig,
    pub robot: RobotConfig,
    pub servo: ServoConfig,
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
    pub integral_limit: f32,
    pub zero_offsets: JointAnglesConfig,
    pub initial_targets: JointAnglesConfig,
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
                initial_pose: InitialPoseConfig {
                    torso: BodyPoseConfig {
                        x: 0.0,
                        y: 1.08,
                        angle: 0.0,
                    },
                    left_thigh: BodyPoseConfig {
                        x: -0.08,
                        y: 0.63,
                        angle: 0.0,
                    },
                    left_shin: BodyPoseConfig {
                        x: -0.17,
                        y: 0.14,
                        angle: 0.0,
                    },
                    right_thigh: BodyPoseConfig {
                        x: 0.08,
                        y: 0.63,
                        angle: 0.0,
                    },
                    right_shin: BodyPoseConfig {
                        x: 0.17,
                        y: 0.14,
                        angle: 0.0,
                    },
                },
            },
            servo: ServoConfig {
                kp: 20.0,
                ki: 0.0,
                kd: 0.0,
                max_torque: 10.0,
                integral_limit: 10.0,
                zero_offsets: JointAnglesConfig {
                    right_hip: -0.15,
                    right_knee: 1.15,
                    left_hip: -0.15,
                    left_knee: 1.15,
                },
                initial_targets: JointAnglesConfig {
                    right_hip: -0.15,
                    right_knee: 1.15,
                    left_hip: -0.15,
                    left_knee: 1.15,
                },
            },
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
