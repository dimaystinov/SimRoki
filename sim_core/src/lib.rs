mod config;
mod ffi;
mod kinematics;

use nalgebra::{point, vector};
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

pub use config::{
    BodyDynamicsConfig, BodyPoseConfig, InitialPoseConfig, JointAnglesConfig, LinkConfig, PhysicsConfig, RlConfig,
    RobotConfig, ServoConfig, SimulationConfig, WalkConfig,
};
use kinematics::{FiveLinkKinematics, LegSide};

const TORSO_UPRIGHT_KP: f32 = 0.0;
const TORSO_UPRIGHT_KD: f32 = 0.0;
const GROUP_GROUND: Group = Group::GROUP_1;
const GROUP_LEFT_LEG: Group = Group::GROUP_2;
const GROUP_RIGHT_LEG: Group = Group::GROUP_3;
const GROUP_TORSO: Group = Group::GROUP_4;
const GROUP_BALL: Group = Group::GROUP_5;
const GROUP_ENVIRONMENT: Group = Group::GROUP_6;
const ROBOT_BALL_RADIUS_M: f32 = 0.2;
const ROBOT_BALL_MASS_KG: f32 = 0.1;
const ROBOT_BALL_CLEARANCE_Y_M: f32 = 0.02;
const GROUND_Y: f32 = 0.0;

fn all_robot_groups() -> Group {
    GROUP_LEFT_LEG | GROUP_RIGHT_LEG | GROUP_TORSO
}

fn robot_world_filter() -> Group {
    GROUP_GROUND | GROUP_BALL | GROUP_ENVIRONMENT
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    Ball,
    Robot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum JointName {
    RightHip,
    RightKnee,
    LeftHip,
    LeftKnee,
}

impl JointName {
    pub const ALL: [JointName; 4] = [
        JointName::RightHip,
        JointName::RightKnee,
        JointName::LeftHip,
        JointName::LeftKnee,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            JointName::RightHip => "right_hip",
            JointName::RightKnee => "right_knee",
            JointName::LeftHip => "left_hip",
            JointName::LeftKnee => "left_knee",
        }
    }
}

impl std::str::FromStr for JointName {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "right_hip" => Ok(Self::RightHip),
            "right_knee" => Ok(Self::RightKnee),
            "left_hip" => Ok(Self::LeftHip),
            "left_knee" => Ok(Self::LeftKnee),
            _ => Err(format!("unknown joint '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyState {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub vx: f32,
    pub vy: f32,
    pub omega: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointState {
    pub angle: f32,
    pub velocity: f32,
    pub target: f32,
    pub torque: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContactState {
    pub left_foot: bool,
    pub right_foot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub time: f32,
    pub mode: String,
    pub scene: SceneKind,
    pub paused: bool,
    pub base: Option<BodyState>,
    pub ball: Option<BodyState>,
    pub joints: BTreeMap<String, JointState>,
    pub link_masses: BTreeMap<String, f32>,
    pub link_lengths: BTreeMap<String, f32>,
    pub servo_zeros: BTreeMap<String, f32>,
    pub contacts: ContactState,
    pub walk_direction: f32,
    pub walk_enabled: bool,
    pub walk_target_speed: f32,
    pub walk_speed: f32,
    pub walk_support_state: String,
    pub walk_state_time: f32,
    pub left_foot: Option<[f32; 2]>,
    pub right_foot: Option<[f32; 2]>,
    pub walk_left_anchor: [f32; 2],
    pub walk_right_anchor: [f32; 2],
    pub walk_left_touchdown_x: f32,
    pub walk_right_touchdown_x: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlObservation {
    pub values: Vec<f32>,
    pub names: Vec<String>,
    pub action_order: Vec<String>,
    pub torso_height: f32,
    pub torso_angle: f32,
    pub base_x: f32,
    pub target_direction: f32,
    pub center_of_mass: Option<[f32; 2]>,
    pub contacts: ContactState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RlRewardBreakdown {
    pub forward_progress: f32,
    pub ball_progress: f32,
    pub alive_bonus: f32,
    pub upright_bonus: f32,
    pub height_bonus: f32,
    pub contact_bonus: f32,
    pub torque_penalty: f32,
    pub action_delta_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStepResult {
    pub observation: RlObservation,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
    pub episode_time: f32,
    pub breakdown: RlRewardBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStepCommand {
    pub action_deg: [f32; 4],
    #[serde(default)]
    pub repeat_steps: Option<u32>,
    #[serde(default)]
    pub direction: Option<f32>,
    #[serde(default)]
    pub residual: Option<bool>,
    #[serde(default)]
    pub walk_enabled: Option<bool>,
    #[serde(default)]
    pub walk_speed_mps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RlResetCommand {
    #[serde(default)]
    pub direction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkDirectionCommand {
    pub direction: f32,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub speed_mps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WalkConfigCommand {
    #[serde(default)]
    pub nominal_speed_mps: Option<f32>,
    #[serde(default)]
    pub max_speed_mps: Option<f32>,
    #[serde(default)]
    pub max_accel_mps2: Option<f32>,
    #[serde(default)]
    pub cycle_frequency_hz: Option<f32>,
    #[serde(default)]
    pub max_cycle_frequency_hz: Option<f32>,
    #[serde(default)]
    pub nominal_step_length_m: Option<f32>,
    #[serde(default)]
    pub step_length_gain: Option<f32>,
    #[serde(default)]
    pub nominal_step_height_m: Option<f32>,
    #[serde(default)]
    pub run_step_height_m: Option<f32>,
    #[serde(default)]
    pub stance_duty_factor: Option<f32>,
    #[serde(default)]
    pub torso_pitch_kp: Option<f32>,
    #[serde(default)]
    pub torso_pitch_kd: Option<f32>,
    #[serde(default)]
    pub hip_upright_gain: Option<f32>,
    #[serde(default)]
    pub hip_upright_damping: Option<f32>,
    #[serde(default)]
    pub torso_upright_limit_rad: Option<f32>,
    #[serde(default)]
    pub torso_forward_lean_per_speed: Option<f32>,
    #[serde(default)]
    pub torso_forward_lean_max_rad: Option<f32>,
    #[serde(default)]
    pub velocity_kp: Option<f32>,
    #[serde(default)]
    pub pelvis_height_target_m: Option<f32>,
    #[serde(default)]
    pub stance_foot_spread_m: Option<f32>,
    #[serde(default)]
    pub foot_separation_min_m: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServoTargets {
    pub right_hip: Option<f32>,
    pub right_knee: Option<f32>,
    pub left_hip: Option<f32>,
    pub left_knee: Option<f32>,
}

impl ServoTargets {
    pub fn get(&self, joint: JointName) -> Option<f32> {
        match joint {
            JointName::RightHip => self.right_hip,
            JointName::RightKnee => self.right_knee,
            JointName::LeftHip => self.left_hip,
            JointName::LeftKnee => self.left_knee,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointCommand {
    pub joint: JointName,
    pub angle: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseCommand {
    pub base: Option<PoseBase>,
    pub joints: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseBase {
    pub x: f32,
    pub y: f32,
    pub yaw: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitCommand {
    pub name: String,
    pub cycle_s: f32,
    pub phases: Vec<GaitPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitPhase {
    pub duration: f32,
    pub joints: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionSequenceCommand {
    pub frames: Vec<[f32; 5]>,
    #[serde(default)]
    pub loop_enabled: bool,
    #[serde(default)]
    pub repeat_delay_ms: f32,
}

#[derive(Debug, Clone)]
struct JointServo {
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    target: f32,
    commanded_target: f32,
    last_torque: f32,
    integral_error: f32,
}

#[derive(Debug, Clone)]
struct RobotHandles {
    torso: RigidBodyHandle,
    left_thigh: RigidBodyHandle,
    left_shin: RigidBodyHandle,
    left_shin_collider: ColliderHandle,
    right_thigh: RigidBodyHandle,
    right_shin: RigidBodyHandle,
    right_shin_collider: ColliderHandle,
    servos: BTreeMap<JointName, JointServo>,
}

#[derive(Debug, Clone)]
struct GaitRuntime {
    phases: Vec<GaitPhase>,
    cycle_s: f32,
    elapsed: f32,
    start_targets: BTreeMap<String, f32>,
}

#[derive(Debug, Clone)]
struct MotionSequenceRuntime {
    frames: Vec<MotionFrame>,
    elapsed: f32,
    start_targets: BTreeMap<String, f32>,
    loop_enabled: bool,
    repeat_delay_s: f32,
    delay_remaining_s: f32,
}

#[derive(Debug, Clone)]
struct MotionFrame {
    duration_s: f32,
    joints: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkLegPhase {
    LoadingResponse,
    MidStance,
    PreSwing,
    InitialSwing,
    MidSwing,
    TerminalSwing,
}

#[derive(Debug, Clone, Copy)]
struct WalkLegTarget {
    target: [f32; 2],
    phase: WalkLegPhase,
    in_stance: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkSupportState {
    DoubleSupportLeft,
    SingleSupportLeft,
    DoubleSupportRight,
    SingleSupportRight,
}

pub struct Simulation {
    pipeline: PhysicsPipeline,
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhaseMultiSap,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    ball: Option<RigidBodyHandle>,
    robot: Option<RobotHandles>,
    scene: SceneKind,
    config: SimulationConfig,
    canonical_initial_pose: InitialPoseConfig,
    canonical_zero_offsets: JointAnglesConfig,
    canonical_initial_targets: JointAnglesConfig,
    time: f32,
    paused: bool,
    accumulator: f32,
    gait: Option<GaitRuntime>,
    motion_sequence: Option<MotionSequenceRuntime>,
    robot_suspended: bool,
    servo_kp: f32,
    servo_ki: f32,
    servo_kd: f32,
    servo_max_torque: f32,
    servo_max_speed_rad_s: f32,
    rl_episode_time: f32,
    rl_last_base_x: f32,
    rl_last_ball_x: f32,
    rl_last_action_deg: [f32; 4],
    rl_residual_enabled: bool,
    rl_residual_action_deg: [f32; 4],
    rl_target_direction: f32,
    directional_walk_enabled: bool,
    walk_target_speed: f32,
    walk_speed: f32,
    walk_phase: f32,
    walk_support_state: WalkSupportState,
    walk_state_time: f32,
    walk_left_anchor: [f32; 2],
    walk_right_anchor: [f32; 2],
    walk_left_swing_start: [f32; 2],
    walk_right_swing_start: [f32; 2],
    walk_left_touchdown_x: f32,
    walk_right_touchdown_x: f32,
    walk_left_in_stance: bool,
    walk_right_in_stance: bool,
    recovery_active: bool,
    recovery_timer: f32,
    kick_active: bool,
    kick_timer: f32,
    kick_support_side: LegSide,
    kick_swing_side: LegSide,
    kick_impulse_used: bool,
    kick_cooldown: f32,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::from_config(SimulationConfig::default())
    }
}

impl Simulation {
    pub fn from_config(config: SimulationConfig) -> Self {
        Self::new_robot_with_config_and_suspension(config, false)
    }

    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let config = SimulationConfig::load_from_file(path)?;
        Ok(Self::from_config(config))
    }

    pub fn save_config_file(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        self.sync_runtime_settings_into_config();
        self.config.save_to_file(path)
    }

    pub fn new_ball() -> Self {
        Self::new_ball_with_config(SimulationConfig::default())
    }

    pub fn new_ball_with_config(config: SimulationConfig) -> Self {
        let mut sim = Self::empty(SceneKind::Ball, config);
        sim.spawn_ground();
        sim.spawn_ball(vector![0.0, 5.0], 0.5, 1.0);
        sim
    }

    pub fn new_robot() -> Self {
        Self::new_robot_with_config_and_suspension(SimulationConfig::default(), false)
    }

    pub fn new_robot_with_suspension(suspended: bool) -> Self {
        Self::new_robot_with_config_and_suspension(SimulationConfig::default(), suspended)
    }

    pub fn new_robot_with_config_and_suspension(config: SimulationConfig, suspended: bool) -> Self {
        let mut sim = Self::empty(SceneKind::Robot, config);
        sim.robot_suspended = suspended;
        sim.spawn_ground();
        sim.spawn_robot();
        sim.refresh_walk_anchors_from_pose();
        sim.spawn_robot_ball();
        sim.sync_rl_trackers();
        sim
    }

    fn empty(scene: SceneKind, config: SimulationConfig) -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = config.physics.dt.max(1.0 / 1000.0);

        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: vector![0.0, config.physics.gravity_y],
            integration_parameters,
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseMultiSap::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            ball: None,
            robot: None,
            scene,
            config: config.clone(),
            canonical_initial_pose: config.robot.initial_pose.clone(),
            canonical_zero_offsets: config.servo.zero_offsets,
            canonical_initial_targets: config.servo.initial_targets,
            time: 0.0,
            paused: false,
            accumulator: 0.0,
            gait: None,
            motion_sequence: None,
            robot_suspended: false,
            servo_kp: config.servo.kp,
            servo_ki: config.servo.ki,
            servo_kd: config.servo.kd,
            servo_max_torque: config.servo.max_torque,
            servo_max_speed_rad_s: config.servo.max_speed_deg_s.to_radians(),
            rl_episode_time: 0.0,
            rl_last_base_x: 0.0,
            rl_last_ball_x: 0.0,
            rl_last_action_deg: [0.0; 4],
            rl_residual_enabled: false,
            rl_residual_action_deg: [0.0; 4],
            rl_target_direction: 1.0,
            directional_walk_enabled: false,
            walk_target_speed: config.walk.max_speed_mps,
            walk_speed: 0.0,
            walk_phase: 0.0,
            walk_support_state: WalkSupportState::DoubleSupportLeft,
            walk_state_time: 0.0,
            walk_left_anchor: [0.0, GROUND_Y],
            walk_right_anchor: [0.0, GROUND_Y],
            walk_left_swing_start: [0.0, GROUND_Y],
            walk_right_swing_start: [0.0, GROUND_Y],
            walk_left_touchdown_x: 0.0,
            walk_right_touchdown_x: 0.0,
            walk_left_in_stance: true,
            walk_right_in_stance: true,
            recovery_active: false,
            recovery_timer: 0.0,
            kick_active: false,
            kick_timer: 0.0,
            kick_support_side: LegSide::Left,
            kick_swing_side: LegSide::Right,
            kick_impulse_used: false,
            kick_cooldown: 0.0,
        }
    }

    fn spawn_ground(&mut self) {
        let ground = self.bodies.insert(
            RigidBodyBuilder::fixed()
                .translation(vector![0.0, -0.1])
                .build(),
        );
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(self.config.physics.ground_half_width, 0.1)
                .collision_groups(InteractionGroups::new(
                    GROUP_GROUND,
                    all_robot_groups() | GROUP_BALL | GROUP_ENVIRONMENT,
                ))
                .friction(self.config.physics.ground_friction)
                .restitution(self.config.physics.ground_restitution)
                .build(),
            ground,
            &mut self.bodies,
        );
    }

    fn spawn_ball(&mut self, translation: Vector<Real>, radius: f32, mass: f32) {
        let area = std::f32::consts::PI * radius * radius;
        let density = if area > f32::EPSILON {
            mass.max(0.0001) / area
        } else {
            1.0
        };
        let handle = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(translation)
                .angular_damping(0.02)
                .linear_damping(0.002)
                .build(),
        );
        let collider = ColliderBuilder::ball(radius)
            .collision_groups(InteractionGroups::new(
                GROUP_BALL,
                all_robot_groups() | GROUP_GROUND | GROUP_BALL | GROUP_ENVIRONMENT,
            ))
            .density(density)
            .friction(0.01)
            .restitution(0.95)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        self.ball = Some(handle);
    }

    fn spawn_robot_ball(&mut self) {
        let translation = self.robot_ball_translation();
        self.spawn_ball(translation, ROBOT_BALL_RADIUS_M, ROBOT_BALL_MASS_KG);
    }

    fn robot_ball_translation(&self) -> Vector<Real> {
        let pose = &self.config.robot.initial_pose;
        let rightmost_x = pose
            .torso
            .x
            .max(pose.left_thigh.x)
            .max(pose.left_shin.x)
            .max(pose.right_thigh.x)
            .max(pose.right_shin.x);
        let leftmost_x = pose
            .torso
            .x
            .min(pose.left_thigh.x)
            .min(pose.left_shin.x)
            .min(pose.right_thigh.x)
            .min(pose.right_shin.x);
        let ball_offset_x = self.config.robot.ball_spawn_offset_x_m.max(ROBOT_BALL_RADIUS_M + 0.05);
        let ball_x = if self.rl_target_direction >= 0.0 {
            rightmost_x + ball_offset_x
        } else {
            leftmost_x - ball_offset_x
        };
        let ball_y = ROBOT_BALL_RADIUS_M + ROBOT_BALL_CLEARANCE_Y_M;
        vector![ball_x, ball_y]
    }

    fn align_ball_with_target_direction(&mut self) {
        let translation = self.robot_ball_translation();
        if let Some(handle) = self.ball {
            if let Some(ball) = self.bodies.get_mut(handle) {
                ball.set_translation(translation, true);
                ball.set_linvel(vector![0.0, 0.0], true);
                ball.set_angvel(0.0, true);
            }
        }
    }

    fn spawn_robot(&mut self) {
        let torso_half_width = self.config.robot.torso.width * 0.5;
        let torso_half_height = self.config.robot.torso.length * 0.5;
        let leg_half_width = self.config.robot.thigh.width * 0.5;
        let thigh_half_height = self.config.robot.thigh.length * 0.5;
        let shin_half_height = self.config.robot.shin.length * 0.5;
        let pose = self.config.robot.initial_pose.clone();
        let (torso, _) = self.spawn_box(
            vector![pose.torso.x, pose.torso.y],
            pose.torso.angle,
            torso_half_width,
            torso_half_height,
            self.config.robot.torso.mass,
            self.config.robot.torso.friction,
            GROUP_TORSO,
            robot_world_filter(),
        );
        let (left_thigh, _) = self.spawn_box(
            vector![pose.left_thigh.x, pose.left_thigh.y],
            pose.left_thigh.angle,
            leg_half_width,
            thigh_half_height,
            self.config.robot.thigh.mass,
            self.config.robot.thigh.friction,
            GROUP_LEFT_LEG,
            GROUP_LEFT_LEG | robot_world_filter(),
        );
        let (left_shin, left_shin_collider) = self.spawn_box(
            vector![pose.left_shin.x, pose.left_shin.y],
            pose.left_shin.angle,
            leg_half_width,
            shin_half_height,
            self.config.robot.shin.mass,
            self.config.robot.shin.friction,
            GROUP_LEFT_LEG,
            GROUP_LEFT_LEG | robot_world_filter(),
        );
        let (right_thigh, _) = self.spawn_box(
            vector![pose.right_thigh.x, pose.right_thigh.y],
            pose.right_thigh.angle,
            leg_half_width,
            thigh_half_height,
            self.config.robot.thigh.mass,
            self.config.robot.thigh.friction,
            GROUP_RIGHT_LEG,
            GROUP_RIGHT_LEG | robot_world_filter(),
        );
        let (right_shin, right_shin_collider) = self.spawn_box(
            vector![pose.right_shin.x, pose.right_shin.y],
            pose.right_shin.angle,
            leg_half_width,
            shin_half_height,
            self.config.robot.shin.mass,
            self.config.robot.shin.friction,
            GROUP_RIGHT_LEG,
            GROUP_RIGHT_LEG | robot_world_filter(),
        );

        self.insert_revolute(
            torso,
            left_thigh,
            point![0.0, -torso_half_height],
            point![0.0, thigh_half_height],
        );
        self.insert_revolute(left_thigh, left_shin, point![0.0, -thigh_half_height], point![0.0, shin_half_height]);
        self.insert_revolute(
            torso,
            right_thigh,
            point![0.0, -torso_half_height],
            point![0.0, thigh_half_height],
        );
        self.insert_revolute(right_thigh, right_shin, point![0.0, -thigh_half_height], point![0.0, shin_half_height]);

        if self.robot_suspended {
            let anchor = self.bodies.insert(
                RigidBodyBuilder::fixed()
                    .translation(vector![
                        pose.torso.x,
                        pose.torso.y + torso_half_height + self.config.robot.suspend_clearance
                    ])
                    .build(),
            );
            let hang_joint = RevoluteJointBuilder::new()
                .local_anchor1(point![0.0, 0.0])
                .local_anchor2(point![0.0, torso_half_height])
                .contacts_enabled(false)
                .build();
            self.impulse_joints.insert(anchor, torso, hang_joint, true);
        }

        let mut servos = BTreeMap::new();
        servos.insert(
            JointName::RightHip,
            JointServo::new(torso, right_thigh, self.config.servo.initial_targets.right_hip),
        );
        servos.insert(
            JointName::RightKnee,
            JointServo::new(right_thigh, right_shin, self.config.servo.initial_targets.right_knee),
        );
        servos.insert(
            JointName::LeftHip,
            JointServo::new(torso, left_thigh, self.config.servo.initial_targets.left_hip),
        );
        servos.insert(
            JointName::LeftKnee,
            JointServo::new(left_thigh, left_shin, self.config.servo.initial_targets.left_knee),
        );

        self.robot = Some(RobotHandles {
            torso,
            left_thigh,
            left_shin,
            left_shin_collider,
            right_thigh,
            right_shin,
            right_shin_collider,
            servos,
        });
    }

    fn spawn_box(
        &mut self,
        translation: Vector<Real>,
        angle: f32,
        half_x: f32,
        half_y: f32,
        mass: f32,
        friction: f32,
        membership: Group,
        filter: Group,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let area = (half_x * 2.0) * (half_y * 2.0);
        let density = if area > f32::EPSILON {
            mass.max(0.0001) / area
        } else {
            1.0
        };
        let handle = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(translation)
                .rotation(angle)
                .angular_damping(self.config.robot.body_dynamics.angular_damping)
                .linear_damping(self.config.robot.body_dynamics.linear_damping)
                .build(),
        );
        let collider = self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(half_x, half_y)
                .collision_groups(InteractionGroups::new(
                    membership,
                    filter,
                ))
                .density(density)
                .friction(friction)
                .restitution(0.02)
                .build(),
            handle,
            &mut self.bodies,
        );
        (handle, collider)
    }

    fn insert_revolute(
        &mut self,
        parent: RigidBodyHandle,
        child: RigidBodyHandle,
        parent_anchor: Point<Real>,
        child_anchor: Point<Real>,
    ) {
        let joint = RevoluteJointBuilder::new()
            .local_anchor1(parent_anchor)
            .local_anchor2(child_anchor)
            .contacts_enabled(false)
            .build();
        self.impulse_joints.insert(parent, child, joint, true);
    }

    pub fn reset_ball(&mut self) {
        *self = match self.scene {
            SceneKind::Ball => Self::new_ball_with_config(self.config.clone()),
            SceneKind::Robot => Self::new_robot_with_config_and_suspension(
                self.config.clone(),
                self.robot_suspended,
            ),
        };
    }

    pub fn reset_robot(&mut self) {
        let gains = self.servo_gains();
        let direction = self.rl_target_direction;
        let walk_enabled = self.directional_walk_enabled;
        let walk_target_speed = self.walk_target_speed;
        *self = Self::new_robot_with_config_and_suspension(self.config.clone(), self.robot_suspended);
        self.rl_target_direction = direction;
        self.directional_walk_enabled = walk_enabled;
        self.walk_target_speed = walk_target_speed;
        self.walk_phase = 0.0;
        self.refresh_walk_anchors_from_pose();
        self.align_ball_with_target_direction();
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        let targets = self.config.servo.initial_targets;
        self.apply_targets(ServoTargets {
            right_hip: Some(targets.right_hip),
            right_knee: Some(targets.right_knee),
            left_hip: Some(targets.left_hip),
            left_knee: Some(targets.left_knee),
        });
        self.sync_rl_trackers();
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn set_scene(&mut self, scene: SceneKind) {
        let direction = self.rl_target_direction;
        let walk_enabled = self.directional_walk_enabled;
        let walk_target_speed = self.walk_target_speed;
        *self = match scene {
            SceneKind::Ball => Self::new_ball_with_config(self.config.clone()),
            SceneKind::Robot => Self::new_robot_with_config_and_suspension(self.config.clone(), self.robot_suspended),
        };
        self.rl_target_direction = direction;
        self.directional_walk_enabled = walk_enabled && scene == SceneKind::Robot;
        self.walk_target_speed = walk_target_speed;
        if scene == SceneKind::Robot {
            self.align_ball_with_target_direction();
        }
        self.sync_rl_trackers();
    }

    pub fn set_robot_suspended(&mut self, suspended: bool) {
        let targets = self.current_targets();
        let gains = self.servo_gains();
        let direction = self.rl_target_direction;
        let walk_enabled = self.directional_walk_enabled;
        let walk_target_speed = self.walk_target_speed;
        *self = Self::new_robot_with_config_and_suspension(self.config.clone(), suspended);
        self.rl_target_direction = direction;
        self.directional_walk_enabled = walk_enabled;
        self.walk_target_speed = walk_target_speed;
        self.align_ball_with_target_direction();
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        self.apply_targets(targets);
        self.sync_rl_trackers();
    }

    pub fn robot_suspended(&self) -> bool {
        self.robot_suspended
    }

    pub fn suspend_clearance(&self) -> f32 {
        self.config.robot.suspend_clearance
    }

    pub fn set_suspend_clearance(&mut self, clearance: f32) {
        let clearance = clearance.clamp(0.05, 3.0);
        self.sync_runtime_settings_into_config();
        self.config.robot.suspend_clearance = clearance;

        if self.scene == SceneKind::Robot {
            let paused = self.paused;
            let config = self.config.clone();
            let suspended = self.robot_suspended;
            let direction = self.rl_target_direction;
            let walk_enabled = self.directional_walk_enabled;
            let walk_target_speed = self.walk_target_speed;
            *self = Self::new_robot_with_config_and_suspension(config, suspended);
            self.rl_target_direction = direction;
            self.directional_walk_enabled = walk_enabled;
            self.walk_target_speed = walk_target_speed;
            self.align_ball_with_target_direction();
            self.paused = paused;
            self.sync_rl_trackers();
        }
    }

    pub fn set_servo_gains(&mut self, kp: f32, ki: f32, kd: f32, max_torque: f32) {
        self.servo_kp = kp.clamp(-20.0, 20.0);
        self.servo_ki = ki.clamp(-5.0, 5.0);
        self.servo_kd = kd.clamp(-1.0, 1.0);
        self.servo_max_torque = max_torque.clamp(0.1, 100.0);
        self.config.servo.kp = self.servo_kp;
        self.config.servo.ki = self.servo_ki;
        self.config.servo.kd = self.servo_kd;
        self.config.servo.max_torque = self.servo_max_torque;
    }

    pub fn servo_gains(&self) -> (f32, f32, f32, f32) {
        (self.servo_kp, self.servo_ki, self.servo_kd, self.servo_max_torque)
    }

    pub fn servo_zero_offsets(&self) -> JointAnglesConfig {
        self.config.servo.zero_offsets
    }

    pub fn set_servo_zero_offsets(&mut self, zeros: JointAnglesConfig) {
        self.config.servo.zero_offsets = zeros;
        self.canonical_zero_offsets = zeros;
    }

    pub fn set_servo_zero_to_current_pose(&mut self) {
        let zeros = JointAnglesConfig {
            right_hip: self
                .robot
                .as_ref()
                .and_then(|robot| self.relative_angle(robot.torso, robot.right_thigh))
                .unwrap_or(self.config.servo.zero_offsets.right_hip),
            right_knee: self
                .robot
                .as_ref()
                .and_then(|robot| self.relative_angle(robot.right_thigh, robot.right_shin))
                .unwrap_or(self.config.servo.zero_offsets.right_knee),
            left_hip: self
                .robot
                .as_ref()
                .and_then(|robot| self.relative_angle(robot.torso, robot.left_thigh))
                .unwrap_or(self.config.servo.zero_offsets.left_hip),
            left_knee: self
                .robot
                .as_ref()
                .and_then(|robot| self.relative_angle(robot.left_thigh, robot.left_shin))
                .unwrap_or(self.config.servo.zero_offsets.left_knee),
        };
        self.set_servo_zero_offsets(zeros);
    }

    pub fn set_joint_target(&mut self, joint: JointName, angle: f32) {
        if let Some(robot) = &mut self.robot {
            if let Some(servo) = robot.servos.get_mut(&joint) {
                servo.commanded_target = clamp_joint_target(joint, angle);
            }
        }
    }

    pub fn apply_targets(&mut self, targets: ServoTargets) {
        for joint in JointName::ALL {
            if let Some(target) = targets.get(joint) {
                self.set_joint_target(joint, target);
            }
        }
    }

    pub fn apply_pose(&mut self, pose: PoseCommand) {
        let mut targets = ServoTargets::default();
        targets.right_hip = pose.joints.get("right_hip").copied();
        targets.right_knee = pose.joints.get("right_knee").copied();
        targets.left_hip = pose.joints.get("left_hip").copied();
        targets.left_knee = pose.joints.get("left_knee").copied();
        self.apply_targets(targets);
    }

    pub fn set_gait(&mut self, gait: GaitCommand) {
        let start_targets = self.current_targets_map();
        self.gait = Some(GaitRuntime {
            phases: gait.phases,
            cycle_s: gait.cycle_s.max(self.dt()),
            elapsed: 0.0,
            start_targets,
        });
        self.motion_sequence = None;
    }

    pub fn set_motion_sequence_deg(&mut self, command: MotionSequenceCommand) {
        let frames = command
            .frames
            .into_iter()
            .map(|frame| MotionFrame {
                duration_s: (frame[0].max(1.0)) / 1000.0,
                joints: BTreeMap::from([
                    ("right_hip".to_owned(), frame[1].to_radians()),
                    ("right_knee".to_owned(), frame[2].to_radians()),
                    ("left_hip".to_owned(), frame[3].to_radians()),
                    ("left_knee".to_owned(), frame[4].to_radians()),
                ]),
            })
            .collect();
        self.motion_sequence = Some(MotionSequenceRuntime {
            frames,
            elapsed: 0.0,
            start_targets: self.current_targets_map(),
            loop_enabled: command.loop_enabled,
            repeat_delay_s: (command.repeat_delay_ms.max(0.0)) / 1000.0,
            delay_remaining_s: 0.0,
        });
        self.gait = None;
    }

    pub fn clear_gait(&mut self) {
        self.gait = None;
        self.motion_sequence = None;
    }

    pub fn rl_reset_with(&mut self, command: RlResetCommand) -> RlStepResult {
        if self.scene != SceneKind::Robot {
            self.set_scene(SceneKind::Robot);
        }
        if self.robot_suspended {
            self.set_robot_suspended(false);
        }
        if let Some(direction) = command.direction {
            self.set_target_direction(direction);
        }
        self.apply_directional_reset_profile();
        self.reset_robot();
        self.resume();
        self.sync_rl_trackers();
        self.rl_step_result(0.0, 0.0, 0.0)
    }

    pub fn rl_reset(&mut self) -> RlStepResult {
        self.rl_reset_with(RlResetCommand::default())
    }

    pub fn rl_observation(&self) -> RlObservation {
        let state = self.state();
        let base = state.base.clone().unwrap_or(BodyState {
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            omega: 0.0,
        });
        let center_of_mass = self.robot_center_of_mass();
        let left_foot = state.left_foot.unwrap_or([base.x, GROUND_Y]);
        let right_foot = state.right_foot.unwrap_or([base.x, GROUND_Y]);
        let joint_names = ["right_hip", "right_knee", "left_hip", "left_knee"];
        let mut values = vec![
            self.rl_target_direction,
            base.y,
            base.angle,
            base.vx,
            base.vy,
            base.omega,
            self.walk_speed,
            self.walk_target_speed,
            self.walk_phase.sin(),
            self.walk_phase.cos(),
            center_of_mass.map(|com| com[0] - base.x).unwrap_or(0.0),
            center_of_mass.map(|com| com[1] - base.y).unwrap_or(0.0),
            if state.contacts.left_foot { 1.0 } else { 0.0 },
            if state.contacts.right_foot { 1.0 } else { 0.0 },
            state.ball.as_ref().map(|ball| ball.x - base.x).unwrap_or(0.0),
            state.ball.as_ref().map(|ball| ball.y - base.y).unwrap_or(0.0),
            left_foot[0] - base.x,
            left_foot[1],
            right_foot[0] - base.x,
            right_foot[1],
        ];
        let mut names = vec![
            "target_direction".to_owned(),
            "torso_height".to_owned(),
            "torso_angle".to_owned(),
            "torso_vx".to_owned(),
            "torso_vy".to_owned(),
            "torso_omega".to_owned(),
            "walk_speed".to_owned(),
            "walk_target_speed".to_owned(),
            "walk_phase_sin".to_owned(),
            "walk_phase_cos".to_owned(),
            "com_dx".to_owned(),
            "com_dy".to_owned(),
            "left_contact".to_owned(),
            "right_contact".to_owned(),
            "ball_dx".to_owned(),
            "ball_dy".to_owned(),
            "left_foot_dx".to_owned(),
            "left_foot_y".to_owned(),
            "right_foot_dx".to_owned(),
            "right_foot_y".to_owned(),
        ];
        for name in joint_names {
            let joint = state.joints.get(name).cloned().unwrap_or(JointState {
                angle: 0.0,
                velocity: 0.0,
                target: 0.0,
                torque: 0.0,
            });
            let zero = state.servo_zeros.get(name).copied().unwrap_or(0.0);
            values.push(normalize_angle(joint.angle - zero));
            names.push(format!("{name}_angle_rel_zero"));
            values.push(joint.velocity);
            names.push(format!("{name}_velocity"));
            values.push(normalize_angle(joint.target - zero));
            names.push(format!("{name}_target_rel_zero"));
            values.push(joint.torque / self.servo_max_torque.max(0.0001));
            names.push(format!("{name}_torque_norm"));
        }

        RlObservation {
            values,
            names,
            action_order: JointName::ALL
                .iter()
                .map(|joint| joint.as_str().to_owned())
                .collect(),
            torso_height: base.y,
            torso_angle: base.angle,
            base_x: base.x,
            target_direction: self.rl_target_direction,
            center_of_mass,
            contacts: state.contacts,
        }
    }

    pub fn rl_step_deg(&mut self, command: RlStepCommand) -> RlStepResult {
        if let Some(direction) = command.direction {
            self.set_target_direction(direction);
        }
        if let Some(enabled) = command.walk_enabled {
            self.set_directional_walk_enabled(enabled);
        }
        if let Some(speed_mps) = command.walk_speed_mps {
            self.set_walk_target_speed(speed_mps);
        }
        self.clear_gait();
        let limit_deg = self.config.rl.action_limit_deg.abs().max(1.0);
        let bounded = command.action_deg.map(|value| value.clamp(-limit_deg, limit_deg));
        let zeros = self.config.servo.zero_offsets;
        let previous_action = self.rl_last_action_deg;
        let base_x_before = self.robot_base_x();
        let ball_x_before = self.ball_x();
        let residual_mode = command.residual.unwrap_or(false);
        self.rl_residual_enabled = residual_mode;
        self.rl_residual_action_deg = if residual_mode { bounded } else { [0.0; 4] };
        if !residual_mode {
            self.apply_targets(ServoTargets {
                right_hip: Some(zeros.right_hip + bounded[0].to_radians()),
                right_knee: Some(zeros.right_knee + bounded[1].to_radians()),
                left_hip: Some(zeros.left_hip + bounded[2].to_radians()),
                left_knee: Some(zeros.left_knee + bounded[3].to_radians()),
            });
        }

        let repeat_steps = command
            .repeat_steps
            .unwrap_or(self.config.rl.control_substeps)
            .max(1);
        let mut executed_steps = 0u32;
        for _ in 0..repeat_steps {
            self.step_fixed();
            executed_steps += 1;
            let (done, truncated) = self.rl_done();
            if done || truncated {
                break;
            }
        }
        self.rl_episode_time += self.dt() * executed_steps as f32;
        let base_x_after = self.robot_base_x();
        let ball_x_after = self.ball_x();
        let forward_progress = base_x_after - base_x_before;
        let ball_progress = (ball_x_after - ball_x_before).max(0.0);
        let action_delta_penalty = bounded
            .iter()
            .zip(previous_action.iter())
            .map(|(current, previous)| (current - previous).abs() / limit_deg)
            .sum::<f32>();
        self.rl_last_action_deg = bounded;
        self.rl_last_base_x = base_x_after;
        self.rl_last_ball_x = ball_x_after;
        self.rl_step_result(forward_progress, ball_progress, action_delta_penalty)
    }

    pub fn set_target_direction(&mut self, direction: f32) {
        self.rl_target_direction = if direction < 0.0 { -1.0 } else { 1.0 };
        if self.scene == SceneKind::Robot {
            self.align_ball_with_target_direction();
        }
    }

    pub fn target_direction(&self) -> f32 {
        self.rl_target_direction
    }

    pub fn set_directional_walk_enabled(&mut self, enabled: bool) {
        let was_enabled = self.directional_walk_enabled;
        self.directional_walk_enabled = enabled;
        if enabled {
            if !was_enabled {
                self.refresh_walk_anchors_from_pose();
            }
        } else {
            if was_enabled {
                self.walk_phase = 0.0;
                self.walk_speed = 0.0;
                self.walk_state_time = 0.0;
            }
        }
    }

    pub fn directional_walk_enabled(&self) -> bool {
        self.directional_walk_enabled
    }

    pub fn set_walk_target_speed(&mut self, speed_mps: f32) {
        self.walk_target_speed = speed_mps.clamp(0.0, self.config.walk.max_speed_mps.max(0.1));
    }

    pub fn walk_target_speed(&self) -> f32 {
        self.walk_target_speed
    }

    pub fn walk_config(&self) -> WalkConfig {
        self.config.walk.clone()
    }

    pub fn update_walk_config(&mut self, command: WalkConfigCommand) {
        if let Some(value) = command.nominal_speed_mps {
            self.config.walk.nominal_speed_mps = value.clamp(0.0, 3.0);
        }
        if let Some(value) = command.max_speed_mps {
            self.config.walk.max_speed_mps = value.clamp(0.1, 4.0);
        }
        if let Some(value) = command.max_accel_mps2 {
            self.config.walk.max_accel_mps2 = value.clamp(0.05, 10.0);
        }
        if let Some(value) = command.cycle_frequency_hz {
            self.config.walk.cycle_frequency_hz = value.clamp(0.2, 6.0);
        }
        if let Some(value) = command.max_cycle_frequency_hz {
            self.config.walk.max_cycle_frequency_hz = value.clamp(0.2, 8.0);
        }
        if let Some(value) = command.nominal_step_length_m {
            self.config.walk.nominal_step_length_m = value.clamp(0.02, 0.8);
        }
        if let Some(value) = command.step_length_gain {
            self.config.walk.step_length_gain = value.clamp(0.0, 1.0);
        }
        if let Some(value) = command.nominal_step_height_m {
            self.config.walk.nominal_step_height_m = value.clamp(0.0, 0.4);
        }
        if let Some(value) = command.run_step_height_m {
            self.config.walk.run_step_height_m = value.clamp(0.0, 0.6);
        }
        if let Some(value) = command.stance_duty_factor {
            self.config.walk.stance_duty_factor = value.clamp(0.50, 0.85);
        }
        if let Some(value) = command.torso_pitch_kp {
            self.config.walk.torso_pitch_kp = value.clamp(0.0, 2.0);
        }
        if let Some(value) = command.torso_pitch_kd {
            self.config.walk.torso_pitch_kd = value.clamp(0.0, 1.0);
        }
        if let Some(value) = command.hip_upright_gain {
            self.config.walk.hip_upright_gain = value.clamp(0.0, 3.0);
        }
        if let Some(value) = command.hip_upright_damping {
            self.config.walk.hip_upright_damping = value.clamp(0.0, 1.0);
        }
        if let Some(value) = command.torso_upright_limit_rad {
            self.config.walk.torso_upright_limit_rad = value.clamp(0.02, 0.5);
        }
        if let Some(value) = command.torso_forward_lean_per_speed {
            self.config.walk.torso_forward_lean_per_speed = value.clamp(0.0, 0.6);
        }
        if let Some(value) = command.torso_forward_lean_max_rad {
            self.config.walk.torso_forward_lean_max_rad = value.clamp(0.0, 0.25);
        }
        if let Some(value) = command.velocity_kp {
            self.config.walk.velocity_kp = value.clamp(0.0, 2.0);
        }
        if let Some(value) = command.pelvis_height_target_m {
            self.config.walk.pelvis_height_target_m = value.clamp(0.4, 1.2);
        }
        if let Some(value) = command.stance_foot_spread_m {
            self.config.walk.stance_foot_spread_m = value.clamp(0.02, 0.5);
        }
        if let Some(value) = command.foot_separation_min_m {
            self.config.walk.foot_separation_min_m = value.clamp(0.02, 0.4);
        }
        self.walk_target_speed = self
            .walk_target_speed
            .clamp(0.0, self.config.walk.max_speed_mps.max(0.1));
    }

    pub fn step_for_seconds(&mut self, frame_dt: f32) {
        self.accumulator += frame_dt.max(0.0);
        while self.accumulator >= self.dt() {
            self.step_fixed();
            self.accumulator -= self.dt();
        }
    }

    pub fn step_fixed(&mut self) {
        if self.paused {
            return;
        }

        if self.scene == SceneKind::Robot {
            self.apply_directional_walk_targets();
            self.apply_motion_sequence_targets();
            self.apply_gait_targets();
            self.apply_rl_residual_targets();
            self.apply_servo_forces();
        }

        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        self.time += self.dt();
    }

    fn apply_gait_targets(&mut self) {
        let dt = self.dt();
        let Some(gait) = &mut self.gait else {
            return;
        };
        if gait.phases.is_empty() {
            return;
        }

        gait.elapsed = (gait.elapsed + dt) % gait.cycle_s;
        let mut elapsed = 0.0;
        let mut phase_index = 0usize;
        let mut phase_start = 0.0f32;
        for (idx, candidate) in gait.phases.iter().enumerate() {
            let duration = candidate.duration.max(dt);
            if gait.elapsed <= elapsed + duration {
                phase_index = idx;
                phase_start = elapsed;
                break;
            }
            elapsed += duration;
        }
        let phase = &gait.phases[phase_index];
        let duration = phase.duration.max(dt);
        let alpha = ((gait.elapsed - phase_start) / duration).clamp(0.0, 1.0);
        let start = if phase_index == 0 {
            &gait.start_targets
        } else {
            &gait.phases[phase_index - 1].joints
        };
        let joints = interpolate_joint_map(start, &phase.joints, alpha);
        let _ = gait;
        self.apply_pose(PoseCommand { base: None, joints });
    }

    fn apply_motion_sequence_targets(&mut self) {
        let dt = self.dt();
        let Some(sequence) = &mut self.motion_sequence else {
            return;
        };
        if sequence.frames.is_empty() {
            self.motion_sequence = None;
            return;
        }

        sequence.elapsed += dt;
        let mut elapsed = 0.0f32;
        let mut phase_index = 0usize;
        let mut phase_start = 0.0f32;
        let total_duration: f32 = sequence.frames.iter().map(|frame| frame.duration_s.max(dt)).sum();

        if sequence.elapsed >= total_duration {
            let final_joints = sequence
                .frames
                .last()
                .map(|frame| frame.joints.clone())
                .unwrap_or_default();
            if sequence.loop_enabled {
                sequence.elapsed = 0.0;
                sequence.delay_remaining_s = sequence.repeat_delay_s;
                sequence.start_targets = final_joints.clone();
            } else {
                self.motion_sequence = None;
                self.apply_pose(PoseCommand {
                    base: None,
                    joints: final_joints,
                });
                return;
            }
        }

        if sequence.delay_remaining_s > 0.0 {
            sequence.delay_remaining_s = (sequence.delay_remaining_s - dt).max(0.0);
            return;
        }

        for (idx, frame) in sequence.frames.iter().enumerate() {
            let duration = frame.duration_s.max(dt);
            if sequence.elapsed <= elapsed + duration {
                phase_index = idx;
                phase_start = elapsed;
                break;
            }
            elapsed += duration;
        }

        let frame = &sequence.frames[phase_index];
        let duration = frame.duration_s.max(dt);
        let alpha = ((sequence.elapsed - phase_start) / duration).clamp(0.0, 1.0);
        let start = if phase_index == 0 {
            &sequence.start_targets
        } else {
            &sequence.frames[phase_index - 1].joints
        };
        let joints = interpolate_joint_map(start, &frame.joints, alpha);
        let _ = sequence;
        self.apply_pose(PoseCommand { base: None, joints });
    }

    fn apply_servo_forces(&mut self) {
        let dt = self.dt();
        let integral_limit = self.config.servo.integral_limit;
        let Some(robot) = &mut self.robot else {
            return;
        };
        let torso_handle = robot.torso;

        for servo in robot.servos.values_mut() {
            let max_target_delta = self.servo_max_speed_rad_s.max(0.01) * dt;
            let target_error = normalize_angle(servo.commanded_target - servo.target);
            let target_step = target_error.clamp(-max_target_delta, max_target_delta);
            servo.target = normalize_angle(servo.target + target_step);

            let (rel_angle, rel_velocity) = match (
                self.bodies.get(servo.parent),
                self.bodies.get(servo.child),
            ) {
                (Some(parent), Some(child)) => (
                    normalize_angle(child.rotation().angle() - parent.rotation().angle()),
                    child.angvel() - parent.angvel(),
                ),
                _ => continue,
            };

            let error = normalize_angle(servo.target - rel_angle);
            servo.integral_error =
                (servo.integral_error + error * dt).clamp(-integral_limit, integral_limit);
            let torque = (self.servo_kp * error
                + self.servo_ki * servo.integral_error
                - self.servo_kd * rel_velocity)
                .clamp(-self.servo_max_torque, self.servo_max_torque);
            servo.last_torque = torque;
            let impulse = torque * dt;

            if let Some(parent) = self.bodies.get_mut(servo.parent) {
                parent.apply_torque_impulse(-impulse, true);
            }
            if let Some(child) = self.bodies.get_mut(servo.child) {
                child.apply_torque_impulse(impulse, true);
            }
        }

        if let Some(torso) = self.bodies.get(torso_handle) {
            let upright_torque =
                (-TORSO_UPRIGHT_KP * torso.rotation().angle() - TORSO_UPRIGHT_KD * torso.angvel())
                    .clamp(-self.servo_max_torque, self.servo_max_torque);
            let impulse = upright_torque * dt;
            if let Some(torso) = self.bodies.get_mut(torso_handle) {
                torso.apply_torque_impulse(impulse, true);
            }
        }
    }

    pub fn state(&self) -> SimulationState {
        let base = self.robot.as_ref().and_then(|robot| self.body_state(robot.torso));
        let ball = self.ball.and_then(|handle| self.body_state(handle));
        let mut joints = BTreeMap::new();
        let mut link_masses = BTreeMap::new();
        let mut link_lengths = BTreeMap::new();
        let mut servo_zeros = BTreeMap::new();

        if let Some(robot) = &self.robot {
            for (name, servo) in &robot.servos {
                joints.insert(
                    name.as_str().to_owned(),
                    JointState {
                        angle: self.relative_angle(servo.parent, servo.child).unwrap_or_default(),
                        velocity: self.relative_velocity(servo.parent, servo.child).unwrap_or_default(),
                        target: servo.target,
                        torque: servo.last_torque,
                    },
                );
            }

            for (name, handle) in [
                ("torso", robot.torso),
                ("left_thigh", robot.left_thigh),
                ("left_shin", robot.left_shin),
                ("right_thigh", robot.right_thigh),
                ("right_shin", robot.right_shin),
            ] {
                if let Some(body) = self.bodies.get(handle) {
                    link_masses.insert(name.to_owned(), body.mass());
                }
            }

            link_lengths.insert("torso".to_owned(), self.config.robot.torso.length);
            link_lengths.insert("left_thigh".to_owned(), self.config.robot.thigh.length);
            link_lengths.insert("left_shin".to_owned(), self.config.robot.shin.length);
            link_lengths.insert("right_thigh".to_owned(), self.config.robot.thigh.length);
            link_lengths.insert("right_shin".to_owned(), self.config.robot.shin.length);

            servo_zeros.insert("right_hip".to_owned(), self.config.servo.zero_offsets.right_hip);
            servo_zeros.insert("right_knee".to_owned(), self.config.servo.zero_offsets.right_knee);
            servo_zeros.insert("left_hip".to_owned(), self.config.servo.zero_offsets.left_hip);
            servo_zeros.insert("left_knee".to_owned(), self.config.servo.zero_offsets.left_knee);
        }

        SimulationState {
            time: self.time,
            mode: if self.paused { "paused" } else { "running" }.to_owned(),
            scene: self.scene,
            paused: self.paused,
            base,
            ball,
            joints,
            link_masses,
            link_lengths,
            servo_zeros,
            contacts: ContactState {
                left_foot: self.foot_contact(true),
                right_foot: self.foot_contact(false),
            },
            walk_direction: self.rl_target_direction,
            walk_enabled: self.directional_walk_enabled,
            walk_target_speed: self.walk_target_speed,
            walk_speed: self.walk_speed,
            walk_support_state: format!("{:?}", self.walk_support_state),
            walk_state_time: self.walk_state_time,
            left_foot: self.foot_world(LegSide::Left),
            right_foot: self.foot_world(LegSide::Right),
            walk_left_anchor: self.walk_left_anchor,
            walk_right_anchor: self.walk_right_anchor,
            walk_left_touchdown_x: self.walk_left_touchdown_x,
            walk_right_touchdown_x: self.walk_right_touchdown_x,
        }
    }

    fn apply_rl_residual_targets(&mut self) {
        if !self.rl_residual_enabled {
            return;
        }
        let current = self.current_commanded_targets();
        self.apply_targets(ServoTargets {
            right_hip: current
                .right_hip
                .map(|value| clamp_joint_target(JointName::RightHip, value + self.rl_residual_action_deg[0].to_radians())),
            right_knee: current
                .right_knee
                .map(|value| clamp_joint_target(JointName::RightKnee, value + self.rl_residual_action_deg[1].to_radians())),
            left_hip: current
                .left_hip
                .map(|value| clamp_joint_target(JointName::LeftHip, value + self.rl_residual_action_deg[2].to_radians())),
            left_knee: current
                .left_knee
                .map(|value| clamp_joint_target(JointName::LeftKnee, value + self.rl_residual_action_deg[3].to_radians())),
        });
    }

    pub fn robot_segments(&self) -> Vec<([f32; 2], [f32; 2])> {
        let Some(robot) = &self.robot else {
            return Vec::new();
        };
        let torso_half_height = self.config.robot.torso.length * 0.5;
        let thigh_half_height = self.config.robot.thigh.length * 0.5;
        let shin_half_height = self.config.robot.shin.length * 0.5;
        let torso_top = self.body_point(robot.torso, point![0.0, torso_half_height]);
        let pelvis = self.body_point(robot.torso, point![0.0, -torso_half_height]);
        let left_knee = self.body_point(robot.left_thigh, point![0.0, -thigh_half_height]);
        let left_foot = self.body_point(robot.left_shin, point![0.0, -shin_half_height]);
        let right_knee = self.body_point(robot.right_thigh, point![0.0, -thigh_half_height]);
        let right_foot = self.body_point(robot.right_shin, point![0.0, -shin_half_height]);

        let points = [torso_top, pelvis, left_knee, left_foot, right_knee, right_foot];
        if points.iter().any(Option::is_none) {
            return Vec::new();
        }
        let points: Vec<[f32; 2]> = points.into_iter().flatten().collect();
        vec![
            (points[1], points[0]),
            (points[1], points[2]),
            (points[2], points[3]),
            (points[1], points[4]),
            (points[4], points[5]),
        ]
    }

    pub fn joint_markers(&self) -> BTreeMap<String, [f32; 2]> {
        let Some(robot) = &self.robot else {
            return BTreeMap::new();
        };
        let mut markers = BTreeMap::new();
        let torso_half_height = self.config.robot.torso.length * 0.5;
        let thigh_half_height = self.config.robot.thigh.length * 0.5;
        if let Some(pelvis) = self.body_point(robot.torso, point![0.0, -torso_half_height]) {
            markers.insert("left_hip".to_owned(), [pelvis[0] - 0.05, pelvis[1]]);
            markers.insert("right_hip".to_owned(), [pelvis[0] + 0.05, pelvis[1]]);
        }
        if let Some(left_knee) = self.body_point(robot.left_thigh, point![0.0, -thigh_half_height]) {
            markers.insert("left_knee".to_owned(), left_knee);
        }
        if let Some(right_knee) = self.body_point(robot.right_thigh, point![0.0, -thigh_half_height]) {
            markers.insert("right_knee".to_owned(), right_knee);
        }
        markers
    }

    pub fn robot_center_of_mass(&self) -> Option<[f32; 2]> {
        let robot = self.robot.as_ref()?;
        let handles = [
            robot.torso,
            robot.left_thigh,
            robot.left_shin,
            robot.right_thigh,
            robot.right_shin,
        ];
        let mut mass_sum = 0.0f32;
        let mut weighted = vector![0.0f32, 0.0f32];
        for handle in handles {
            let body = self.bodies.get(handle)?;
            let mass = body.mass();
            let pos = body.translation();
            weighted += vector![pos.x * mass, pos.y * mass];
            mass_sum += mass;
        }
        if mass_sum <= f32::EPSILON {
            None
        } else {
            Some([weighted.x / mass_sum, weighted.y / mass_sum])
        }
    }

    fn robot_kinematics(&self) -> FiveLinkKinematics {
        FiveLinkKinematics {
            torso_length: self.config.robot.torso.length,
            thigh_length: self.config.robot.thigh.length,
            shin_length: self.config.robot.shin.length,
        }
    }

    fn pelvis_world(&self) -> Option<[f32; 2]> {
        let robot = self.robot.as_ref()?;
        let torso_half_height = self.config.robot.torso.length * 0.5;
        self.body_point(robot.torso, point![0.0, -torso_half_height])
    }

    fn foot_world(&self, side: LegSide) -> Option<[f32; 2]> {
        let robot = self.robot.as_ref()?;
        let shin_half_height = self.config.robot.shin.length * 0.5;
        match side {
            LegSide::Left => self.body_point(robot.left_shin, point![0.0, -shin_half_height]),
            LegSide::Right => self.body_point(robot.right_shin, point![0.0, -shin_half_height]),
        }
    }

    fn ball_body_state(&self) -> Option<BodyState> {
        self.ball.and_then(|handle| self.body_state(handle))
    }

    fn refresh_walk_anchors_from_pose(&mut self) {
        if let Some(left) = self.foot_world(LegSide::Left) {
            self.walk_left_anchor = [left[0], GROUND_Y];
            self.walk_left_swing_start = left;
            self.walk_left_touchdown_x = left[0];
        }
        if let Some(right) = self.foot_world(LegSide::Right) {
            self.walk_right_anchor = [right[0], GROUND_Y];
            self.walk_right_swing_start = right;
            self.walk_right_touchdown_x = right[0];
        }
        self.walk_support_state = if opposite_leg(self.front_leg_side(self.rl_target_direction)) == LegSide::Left {
            WalkSupportState::DoubleSupportLeft
        } else {
            WalkSupportState::DoubleSupportRight
        };
        self.walk_state_time = 0.0;
        self.sync_walk_support_flags();
        self.recovery_active = false;
        self.recovery_timer = 0.0;
        self.kick_active = false;
        self.kick_timer = 0.0;
        self.kick_impulse_used = false;
        self.kick_cooldown = 0.0;
    }

    fn sync_walk_support_flags(&mut self) {
        match self.walk_support_state {
            WalkSupportState::DoubleSupportLeft | WalkSupportState::DoubleSupportRight => {
                self.walk_left_in_stance = true;
                self.walk_right_in_stance = true;
            }
            WalkSupportState::SingleSupportLeft => {
                self.walk_left_in_stance = true;
                self.walk_right_in_stance = false;
            }
            WalkSupportState::SingleSupportRight => {
                self.walk_left_in_stance = false;
                self.walk_right_in_stance = true;
            }
        }
    }

    fn walk_state_durations(&self, cycle_hz: f32, speed_ratio: f32) -> (f32, f32) {
        let half_step_duration = 0.5 / cycle_hz.max(0.2);
        let double_support_fraction =
            ((self.config.walk.stance_duty_factor - 0.5) * 0.7 + 0.10 * (1.0 - speed_ratio)).clamp(0.16, 0.32);
        let double_support_duration = (half_step_duration * double_support_fraction).max(self.dt() * 2.0);
        let single_support_duration = (half_step_duration - double_support_duration).max(self.dt() * 4.0);
        (double_support_duration, single_support_duration)
    }

    fn update_walk_phase_from_state(&mut self, double_support_duration: f32, single_support_duration: f32) {
        let total_cycle = 2.0 * (double_support_duration + single_support_duration);
        let phase_offset = match self.walk_support_state {
            WalkSupportState::DoubleSupportLeft => 0.0,
            WalkSupportState::SingleSupportLeft => double_support_duration,
            WalkSupportState::DoubleSupportRight => double_support_duration + single_support_duration,
            WalkSupportState::SingleSupportRight => 2.0 * double_support_duration + single_support_duration,
        };
        self.walk_phase = std::f32::consts::TAU * ((phase_offset + self.walk_state_time) / total_cycle.max(self.dt()));
    }

    fn current_walk_state_duration(&self, double_support_duration: f32, single_support_duration: f32) -> f32 {
        match self.walk_support_state {
            WalkSupportState::DoubleSupportLeft | WalkSupportState::DoubleSupportRight => double_support_duration,
            WalkSupportState::SingleSupportLeft | WalkSupportState::SingleSupportRight => single_support_duration,
        }
    }

    fn capture_swing_start(&mut self, side: LegSide) {
        let start = self.foot_world(side).unwrap_or_else(|| match side {
            LegSide::Left => self.walk_left_anchor,
            LegSide::Right => self.walk_right_anchor,
        });
        match side {
            LegSide::Left => self.walk_left_swing_start = start,
            LegSide::Right => self.walk_right_swing_start = start,
        }
    }

    fn plan_walk_touchdown_x(
        &self,
        support_x: f32,
        direction: f32,
        step_length: f32,
        support_shift: f32,
        spread: f32,
    ) -> f32 {
        let local_step_advance = (1.10 * step_length + 0.55 * spread + 0.35 * support_shift).clamp(0.06, 0.22);
        support_x + direction * local_step_advance
    }

    fn begin_single_support(
        &mut self,
        support_side: LegSide,
        direction: f32,
        step_length: f32,
        support_shift: f32,
        spread: f32,
    ) {
        let swing_side = opposite_leg(support_side);
        self.capture_swing_start(swing_side);
        let support_x = self
            .foot_world(support_side)
            .map(|foot| foot[0])
            .unwrap_or_else(|| match support_side {
                LegSide::Left => self.walk_left_anchor[0],
                LegSide::Right => self.walk_right_anchor[0],
            });
        let touchdown_x = self.plan_walk_touchdown_x(support_x, direction, step_length, support_shift, spread);
        match swing_side {
            LegSide::Left => self.walk_left_touchdown_x = touchdown_x,
            LegSide::Right => self.walk_right_touchdown_x = touchdown_x,
        }
        self.walk_support_state = match support_side {
            LegSide::Left => WalkSupportState::SingleSupportLeft,
            LegSide::Right => WalkSupportState::SingleSupportRight,
        };
        self.walk_state_time = 0.0;
        self.sync_walk_support_flags();
    }

    fn begin_double_support(&mut self, lead_side: LegSide) {
        match lead_side {
            LegSide::Left => {
                self.walk_left_anchor = [self.walk_left_touchdown_x, GROUND_Y];
                self.walk_left_swing_start = self.walk_left_anchor;
                self.walk_support_state = WalkSupportState::DoubleSupportLeft;
            }
            LegSide::Right => {
                self.walk_right_anchor = [self.walk_right_touchdown_x, GROUND_Y];
                self.walk_right_swing_start = self.walk_right_anchor;
                self.walk_support_state = WalkSupportState::DoubleSupportRight;
            }
        }
        self.walk_state_time = 0.0;
        self.sync_walk_support_flags();
    }

    fn advance_walk_support_state(
        &mut self,
        direction: f32,
        step_length: f32,
        support_shift: f32,
        spread: f32,
    ) {
        match self.walk_support_state {
            WalkSupportState::DoubleSupportLeft => {
                self.begin_single_support(LegSide::Left, direction, step_length, support_shift, spread);
            }
            WalkSupportState::SingleSupportLeft => {
                self.begin_double_support(LegSide::Right);
            }
            WalkSupportState::DoubleSupportRight => {
                self.begin_single_support(LegSide::Right, direction, step_length, support_shift, spread);
            }
            WalkSupportState::SingleSupportRight => {
                self.begin_double_support(LegSide::Left);
            }
        }
    }

    fn apply_directional_walk_targets(&mut self) {
        if !self.directional_walk_enabled || self.motion_sequence.is_some() || self.gait.is_some() {
            return;
        }

        let dt = self.dt();
        let direction = self.rl_target_direction;
        let Some(robot) = &self.robot else {
            return;
        };
        let Some(base) = self.body_state(robot.torso) else {
            return;
        };
        let Some(pelvis) = self.pelvis_world() else {
            return;
        };
        let kinematics = self.robot_kinematics();
        let walk_cfg = self.config.walk.clone();
        self.kick_cooldown = (self.kick_cooldown - dt).max(0.0);

        if base.y < 0.45 || base.angle.abs() > 1.6 {
            self.emergency_stand_reset();
            return;
        }

        if self.recovery_active
            || base.y < walk_cfg.recovery_height_threshold_m
            || base.angle.abs() > walk_cfg.recovery_angle_threshold_rad
        {
            if !self.recovery_active {
                self.recovery_active = true;
                self.recovery_timer = 0.0;
                self.kick_active = false;
                self.kick_timer = 0.0;
                self.kick_impulse_used = false;
            }
            self.apply_recovery_targets(base.angle, base.y, dt);
            if base.y > walk_cfg.recovery_height_threshold_m + 0.18 && base.angle.abs() < 0.45 {
                self.recovery_active = false;
                self.recovery_timer = 0.0;
                self.walk_phase = 0.0;
                self.refresh_walk_anchors_from_pose();
            } else if self.recovery_timer > 2.2 {
                self.emergency_stand_reset();
            }
            return;
        }

        let forward_velocity = base.vx * direction;
        let max_accel = walk_cfg.max_accel_mps2.max(0.05);
        let speed_error = self.walk_target_speed - self.walk_speed;
        let speed_step = max_accel * dt;
        self.walk_speed += speed_error.clamp(-speed_step, speed_step);

        let speed_ratio = (self.walk_speed / walk_cfg.max_speed_mps.max(0.1)).clamp(0.0, 1.0);
        let cycle_hz = (walk_cfg.cycle_frequency_hz
            + (walk_cfg.max_cycle_frequency_hz - walk_cfg.cycle_frequency_hz) * speed_ratio)
            * (0.90 + 0.10 * speed_ratio);
        let (double_support_duration, single_support_duration) = self.walk_state_durations(cycle_hz, speed_ratio);
        self.walk_state_time += dt;

        let lean_sign = if forward_velocity.abs() > 0.05 {
            forward_velocity.signum()
        } else {
            direction
        };
        let desired_torso_angle = (lean_sign
            * (self.walk_speed * walk_cfg.torso_forward_lean_per_speed)
                .clamp(0.0, walk_cfg.torso_forward_lean_max_rad))
            .clamp(-walk_cfg.torso_upright_limit_rad, walk_cfg.torso_upright_limit_rad);
        let torso_angle_error = (base.angle - desired_torso_angle)
            .clamp(-walk_cfg.torso_upright_limit_rad, walk_cfg.torso_upright_limit_rad);
        let torso_feedback =
            (-walk_cfg.torso_pitch_kp * torso_angle_error - walk_cfg.torso_pitch_kd * base.omega).clamp(-0.16, 0.16);
        let velocity_feedback =
            (walk_cfg.velocity_kp * (self.walk_speed - forward_velocity)).clamp(-0.20, 0.20);
        let support_shift = (torso_feedback + velocity_feedback) * 0.06;
        let step_length = (walk_cfg.nominal_step_length_m
            + walk_cfg.step_length_gain * self.walk_speed
            + 0.008 * speed_ratio)
            .clamp(0.04, 0.18);
        let step_height = (walk_cfg.nominal_step_height_m
            + (walk_cfg.run_step_height_m - walk_cfg.nominal_step_height_m) * 0.30 * speed_ratio)
            .clamp(0.018, 0.065);
        let swing_retraction_m = (0.012 + 0.022 * speed_ratio).clamp(0.012, 0.036);
        let spread = walk_cfg.stance_foot_spread_m.clamp(0.08, 0.30);

        while self.walk_state_time
            >= self.current_walk_state_duration(double_support_duration, single_support_duration)
        {
            self.walk_state_time -= self.current_walk_state_duration(double_support_duration, single_support_duration);
            self.advance_walk_support_state(direction, step_length, support_shift, spread);
        }
        self.update_walk_phase_from_state(double_support_duration, single_support_duration);

        if !self.kick_active {
            if let Some(ball) = self.ball_body_state() {
                let ball_rel_x = (ball.x - pelvis[0]) * direction;
                let front_side = self.front_leg_side(direction);
                let both_feet_supported = self.foot_contact(true) && self.foot_contact(false);
                if ball_rel_x > 0.32
                    && ball_rel_x < walk_cfg.kick_trigger_distance_m
                    && ball.y < 0.45
                    && base.angle.abs() < 0.32
                    && self.walk_speed > 0.22
                    && both_feet_supported
                    && self.kick_cooldown <= 0.0
                {
                    self.kick_active = true;
                    self.kick_timer = 0.0;
                    self.kick_impulse_used = false;
                    self.kick_swing_side = front_side;
                    self.kick_support_side = match front_side {
                        LegSide::Left => LegSide::Right,
                        LegSide::Right => LegSide::Left,
                    };
                    if let Some(support_foot) = self.foot_world(self.kick_support_side) {
                        match self.kick_support_side {
                            LegSide::Left => self.walk_left_anchor = [support_foot[0], GROUND_Y],
                            LegSide::Right => self.walk_right_anchor = [support_foot[0], GROUND_Y],
                        }
                    }
                }
            }
        }

        if self.kick_active {
            self.kick_timer += dt;
            self.apply_kick_targets(&kinematics, pelvis, base.angle, direction);
            return;
        }

        let state_progress = (self.walk_state_time
            / self.current_walk_state_duration(double_support_duration, single_support_duration).max(self.dt()))
            .clamp(0.0, 1.0);
        let (mut left_leg, mut right_leg) = self.walk_targets_from_state(
            direction,
            state_progress,
            step_length,
            step_height,
            spread,
            support_shift,
            swing_retraction_m,
        );

        let double_support = matches!(
            self.walk_support_state,
            WalkSupportState::DoubleSupportLeft | WalkSupportState::DoubleSupportRight
        );
        let stance_compression = if double_support { 0.014 } else { 0.002 };
        let pelvis_bob = if double_support {
            -0.006 * smoothstep01(state_progress)
        } else {
            0.006 * (1.0 - (2.0 * state_progress - 1.0).abs())
        };
        let desired_pelvis_height = walk_cfg.pelvis_height_target_m + pelvis_bob - stance_compression;
        let vertical_feedback = ((desired_pelvis_height - base.y) * 0.40 - 0.08 * base.vy).clamp(-0.035, 0.035);

        right_leg.target[1] -= vertical_feedback;
        left_leg.target[1] -= vertical_feedback;

        if matches!(right_leg.phase, WalkLegPhase::TerminalSwing) {
            right_leg.target[1] = right_leg.target[1].max(GROUND_Y + 0.006);
        }
        if matches!(left_leg.phase, WalkLegPhase::TerminalSwing) {
            left_leg.target[1] = left_leg.target[1].max(GROUND_Y + 0.006);
        }

        let stance_balance = match self.walk_support_state {
            WalkSupportState::SingleSupportLeft => -1.0,
            WalkSupportState::SingleSupportRight => 1.0,
            WalkSupportState::DoubleSupportLeft => -0.35,
            WalkSupportState::DoubleSupportRight => 0.35,
        };
        let stance_hip_bias = stance_balance * direction * (0.025 + 0.025 * speed_ratio);

        enforce_min_foot_separation(&mut left_leg.target, &mut right_leg.target, walk_cfg.foot_separation_min_m);

        let Some(right_ik) = kinematics.solve_leg_ik(LegSide::Right, pelvis, base.angle, right_leg.target) else {
            return;
        };
        let Some(left_ik) = kinematics.solve_leg_ik(LegSide::Left, pelvis, base.angle, left_leg.target) else {
            return;
        };
        let hip_upright_correction =
            (-walk_cfg.hip_upright_gain * torso_angle_error - walk_cfg.hip_upright_damping * base.omega).clamp(-0.18, 0.18);

        self.apply_targets(ServoTargets {
            right_hip: Some(clamp_joint_target(
                JointName::RightHip,
                right_ik.hip + hip_upright_correction - stance_hip_bias,
            )),
            right_knee: Some(right_ik.knee),
            left_hip: Some(clamp_joint_target(
                JointName::LeftHip,
                left_ik.hip + hip_upright_correction + stance_hip_bias,
            )),
            left_knee: Some(left_ik.knee),
        });
    }

    fn apply_recovery_targets(&mut self, torso_angle: f32, torso_height: f32, dt: f32) {
        self.recovery_timer += dt;
        self.walk_speed = 0.0;
        let zeros = self.config.servo.zero_offsets;
        let roll = if torso_angle >= 0.0 { 1.0 } else { -1.0 };
        let (right_hip, right_knee, left_hip, left_knee) = if torso_height < 0.38 || torso_angle.abs() > 1.6 {
            (
                zeros.right_hip - 0.9 * roll,
                zeros.right_knee + 1.0,
                zeros.left_hip - 0.9 * roll,
                zeros.left_knee - 1.0,
            )
        } else if self.recovery_timer < 0.7 {
            (
                zeros.right_hip - 0.35 * roll,
                zeros.right_knee + 0.75,
                zeros.left_hip - 0.35 * roll,
                zeros.left_knee - 0.75,
            )
        } else if self.recovery_timer < 1.4 {
            (
                zeros.right_hip - 0.15 * roll,
                zeros.right_knee + 0.35,
                zeros.left_hip - 0.15 * roll,
                zeros.left_knee - 0.35,
            )
        } else {
            (zeros.right_hip, zeros.right_knee, zeros.left_hip, zeros.left_knee)
        };
        self.apply_targets(ServoTargets {
            right_hip: Some(right_hip),
            right_knee: Some(right_knee),
            left_hip: Some(left_hip),
            left_knee: Some(left_knee),
        });
    }

    fn front_leg_side(&self, direction: f32) -> LegSide {
        let left_x = self.foot_world(LegSide::Left).map(|p| p[0]).unwrap_or(0.0) * direction;
        let right_x = self.foot_world(LegSide::Right).map(|p| p[0]).unwrap_or(0.0) * direction;
        if right_x >= left_x {
            LegSide::Right
        } else {
            LegSide::Left
        }
    }

    fn apply_kick_targets(
        &mut self,
        kinematics: &FiveLinkKinematics,
        pelvis: [f32; 2],
        torso_angle: f32,
        direction: f32,
    ) {
        let Some(ball) = self.ball_body_state() else {
            self.kick_active = false;
            return;
        };
        let support_target = match self.kick_support_side {
            LegSide::Left => self.walk_left_anchor,
            LegSide::Right => self.walk_right_anchor,
        };
        let support_solution = kinematics.solve_leg_ik(self.kick_support_side, pelvis, torso_angle, support_target);

        let t = (self.kick_timer / 0.42).clamp(0.0, 1.0);
        let swing_start = self.foot_world(self.kick_swing_side).unwrap_or([pelvis[0], GROUND_Y + 0.05]);
        let backswing_x = pelvis[0] - direction * 0.22;
        let backswing_y = GROUND_Y + 0.10;
        let strike_x = ball.x + direction * 0.06;
        let strike_y = (ball.y + 0.02).clamp(GROUND_Y + 0.08, 0.32);
        let swing_target = if t < 0.35 {
            let u = t / 0.35;
            [
                swing_start[0] + (backswing_x - swing_start[0]) * u,
                swing_start[1] + (backswing_y - swing_start[1]) * u,
            ]
        } else {
            let u = (t - 0.35) / 0.65;
            [
                backswing_x + (strike_x - backswing_x) * u,
                backswing_y + (strike_y - backswing_y) * u,
            ]
        };
        let swing_solution = kinematics.solve_leg_ik(self.kick_swing_side, pelvis, torso_angle, swing_target);

        let zeros = self.config.servo.zero_offsets;
        let mut targets = ServoTargets::default();
        match (self.kick_support_side, support_solution) {
            (LegSide::Left, Some(solution)) => {
                targets.left_hip = Some(solution.hip);
                targets.left_knee = Some(solution.knee);
            }
            (LegSide::Right, Some(solution)) => {
                targets.right_hip = Some(solution.hip);
                targets.right_knee = Some(solution.knee);
            }
            (LegSide::Left, None) => {
                targets.left_hip = Some(zeros.left_hip);
                targets.left_knee = Some(zeros.left_knee);
            }
            (LegSide::Right, None) => {
                targets.right_hip = Some(zeros.right_hip);
                targets.right_knee = Some(zeros.right_knee);
            }
        }
        match (self.kick_swing_side, swing_solution) {
            (LegSide::Left, Some(solution)) => {
                targets.left_hip = Some(solution.hip);
                targets.left_knee = Some(solution.knee);
            }
            (LegSide::Right, Some(solution)) => {
                targets.right_hip = Some(solution.hip);
                targets.right_knee = Some(solution.knee);
            }
            _ => {}
        }
        self.apply_targets(targets);

        if !self.kick_impulse_used {
            let ball_rel_x = (ball.x - pelvis[0]) * direction;
            if ball_rel_x > -0.05 && ball_rel_x <= self.config.walk.kick_trigger_distance_m && ball.y <= 0.45 && t > 0.2 {
                if let Some(ball_handle) = self.ball {
                    if let Some(ball_body) = self.bodies.get_mut(ball_handle) {
                        let impulse = vector![
                            direction * self.config.walk.kick_forward_impulse_ns,
                            self.config.walk.kick_upward_impulse_ns
                        ];
                        ball_body.apply_impulse(impulse, true);
                    }
                }
                self.kick_impulse_used = true;
                self.kick_cooldown = 1.2;
            }
        }

        if self.kick_timer >= 0.42 {
            self.kick_active = false;
            self.kick_timer = 0.0;
            self.kick_impulse_used = false;
            self.refresh_walk_anchors_from_pose();
        }
    }

    fn emergency_stand_reset(&mut self) {
        let Some(robot) = &self.robot else {
            return;
        };
        let torso = robot.torso;
        let left_thigh = robot.left_thigh;
        let left_shin = robot.left_shin;
        let right_thigh = robot.right_thigh;
        let right_shin = robot.right_shin;
        let pose = if self.rl_target_direction >= 0.0 {
            self.canonical_initial_pose.clone()
        } else {
            mirror_initial_pose(&self.canonical_initial_pose)
        };
        let targets = if self.rl_target_direction >= 0.0 {
            self.canonical_initial_targets
        } else {
            mirror_joint_angles(self.canonical_initial_targets)
        };
        let base_x = self.body_state(torso).map(|body| body.x).unwrap_or(0.0);
        let shift_x = base_x - pose.torso.x;
        self.set_body_pose(torso, pose.torso, shift_x);
        self.set_body_pose(left_thigh, pose.left_thigh, shift_x);
        self.set_body_pose(left_shin, pose.left_shin, shift_x);
        self.set_body_pose(right_thigh, pose.right_thigh, shift_x);
        self.set_body_pose(right_shin, pose.right_shin, shift_x);
        self.apply_targets(ServoTargets {
            right_hip: Some(targets.right_hip),
            right_knee: Some(targets.right_knee),
            left_hip: Some(targets.left_hip),
            left_knee: Some(targets.left_knee),
        });
        self.recovery_active = false;
        self.recovery_timer = 0.0;
        self.walk_speed = 0.0;
        self.walk_phase = 0.0;
        self.refresh_walk_anchors_from_pose();
    }

    fn apply_directional_reset_profile(&mut self) {
        if self.rl_target_direction >= 0.0 {
            self.config.robot.initial_pose = self.canonical_initial_pose.clone();
            self.config.servo.zero_offsets = self.canonical_zero_offsets;
            self.config.servo.initial_targets = self.canonical_initial_targets;
        } else {
            self.config.robot.initial_pose = mirror_initial_pose(&self.canonical_initial_pose);
            self.config.servo.zero_offsets = mirror_joint_angles(self.canonical_zero_offsets);
            self.config.servo.initial_targets = mirror_joint_angles(self.canonical_initial_targets);
        }
    }

    fn walk_targets_from_state(
        &self,
        direction: f32,
        state_progress: f32,
        step_length: f32,
        step_height: f32,
        spread: f32,
        support_shift: f32,
        swing_retraction_m: f32,
    ) -> (WalkLegTarget, WalkLegTarget) {
        let state_progress = state_progress.clamp(0.0, 1.0);
        let toeoff_blend = smoothstep01(state_progress);
        let toeoff_shift = direction * (0.04 * step_length + 0.10 * spread + 0.15 * support_shift);
        let toe_lift = 0.20 * step_height * toeoff_blend;

        let mut left = WalkLegTarget {
            target: self.walk_left_anchor,
            phase: WalkLegPhase::MidStance,
            in_stance: true,
        };
        let mut right = WalkLegTarget {
            target: self.walk_right_anchor,
            phase: WalkLegPhase::MidStance,
            in_stance: true,
        };

        match self.walk_support_state {
            WalkSupportState::DoubleSupportLeft => {
                left.phase = WalkLegPhase::LoadingResponse;
                right.phase = WalkLegPhase::PreSwing;
                right.target = [
                    lerp(self.walk_right_anchor[0], self.walk_right_anchor[0] + toeoff_shift, 0.35 * toeoff_blend),
                    GROUND_Y + toe_lift,
                ];
            }
            WalkSupportState::SingleSupportLeft => {
                left.phase = WalkLegPhase::MidStance;
                right = self.swing_leg_target(
                    self.walk_right_swing_start,
                    self.walk_right_touchdown_x,
                    direction,
                    state_progress,
                    step_height,
                    swing_retraction_m,
                );
            }
            WalkSupportState::DoubleSupportRight => {
                right.phase = WalkLegPhase::LoadingResponse;
                left.phase = WalkLegPhase::PreSwing;
                left.target = [
                    lerp(self.walk_left_anchor[0], self.walk_left_anchor[0] + toeoff_shift, 0.35 * toeoff_blend),
                    GROUND_Y + toe_lift,
                ];
            }
            WalkSupportState::SingleSupportRight => {
                right.phase = WalkLegPhase::MidStance;
                left = self.swing_leg_target(
                    self.walk_left_swing_start,
                    self.walk_left_touchdown_x,
                    direction,
                    state_progress,
                    step_height,
                    swing_retraction_m,
                );
            }
        }

        (left, right)
    }

    fn swing_leg_target(
        &self,
        swing_start: [f32; 2],
        touchdown_x: f32,
        direction: f32,
        state_progress: f32,
        step_height: f32,
        swing_retraction_m: f32,
    ) -> WalkLegTarget {
        let state_progress = state_progress.clamp(0.0, 1.0);
        let overshoot_x = touchdown_x + direction * swing_retraction_m;
        let (phase, x, y) = if state_progress < 0.32 {
            let u = smoothstep01(state_progress / 0.32);
            (
                WalkLegPhase::InitialSwing,
                lerp(swing_start[0], 0.5 * (swing_start[0] + overshoot_x), u),
                lerp(swing_start[1], GROUND_Y + 0.80 * step_height, u),
            )
        } else if state_progress < 0.72 {
            let u = smoothstep01((state_progress - 0.32) / 0.40);
            let midswing_x = 0.5 * (swing_start[0] + overshoot_x);
            (
                WalkLegPhase::MidSwing,
                lerp(midswing_x, overshoot_x, u),
                GROUND_Y + step_height * (0.85 + 0.15 * (1.0 - (2.0 * u - 1.0).abs())),
            )
        } else {
            let u = smoothstep01((state_progress - 0.72) / 0.28);
            (
                WalkLegPhase::TerminalSwing,
                lerp(overshoot_x, touchdown_x, u),
                lerp(GROUND_Y + 0.55 * step_height, GROUND_Y, u),
            )
        };

        WalkLegTarget {
            target: [x, y],
            phase,
            in_stance: false,
        }
    }

    fn walk_leg_target_world(
        &mut self,
        side: LegSide,
        direction: f32,
        phase: f32,
        pelvis_x: f32,
        step_length: f32,
        step_height: f32,
        spread: f32,
        support_shift: f32,
        stance_duty: f32,
        swing_retraction_m: f32,
    ) -> WalkLegTarget {
        let phase_offset = match side {
            LegSide::Right => 0.0,
            LegSide::Left => std::f32::consts::PI,
        };
        let cycle = ((phase + phase_offset).rem_euclid(std::f32::consts::TAU)) / std::f32::consts::TAU;
        let stance_duty = stance_duty.clamp(0.50, 0.85);
        let current_foot = self.foot_world(side);
        let (phase_kind, phase_progress, in_stance) = classify_walk_leg_phase(cycle, stance_duty);
        let anchor = match side {
            LegSide::Left => &mut self.walk_left_anchor,
            LegSide::Right => &mut self.walk_right_anchor,
        };
        let touchdown_x_slot = match side {
            LegSide::Left => &mut self.walk_left_touchdown_x,
            LegSide::Right => &mut self.walk_right_touchdown_x,
        };
        let was_in_stance = match side {
            LegSide::Left => &mut self.walk_left_in_stance,
            LegSide::Right => &mut self.walk_right_in_stance,
        };
        let touchdown_x = pelvis_x + direction * (0.30 * step_length + 0.45 * spread + support_shift);
        let toeoff_x = anchor[0] + direction * (0.10 * step_length + 0.18 * spread + 0.25 * support_shift);
        let midswing_x = pelvis_x + direction * (0.12 * step_length + 0.22 * spread + 0.35 * support_shift);

        if !in_stance && *was_in_stance {
            if let Some(current_foot) = current_foot {
                *anchor = [current_foot[0], GROUND_Y];
            }
            *touchdown_x_slot = touchdown_x;
        } else if in_stance && !*was_in_stance {
            *anchor = [*touchdown_x_slot, GROUND_Y];
        }
        *was_in_stance = in_stance;

        let touchdown_x = *touchdown_x_slot;
        let overshoot_x = touchdown_x + direction * swing_retraction_m;
        let target = match phase_kind {
            WalkLegPhase::LoadingResponse | WalkLegPhase::MidStance => *anchor,
            WalkLegPhase::PreSwing => {
                let s = smoothstep01(phase_progress);
                [
                    lerp(anchor[0], toeoff_x, 0.35 * s),
                    GROUND_Y + 0.22 * step_height * s,
                ]
            }
            WalkLegPhase::InitialSwing => {
                let s = smoothstep01(phase_progress);
                [
                    lerp(toeoff_x, midswing_x, s),
                    GROUND_Y + lerp(0.22 * step_height, 0.95 * step_height, s),
                ]
            }
            WalkLegPhase::MidSwing => {
                let s = smoothstep01(phase_progress);
                let arch = 1.0 - (2.0 * s - 1.0).abs();
                [
                    lerp(midswing_x, overshoot_x, s),
                    GROUND_Y + step_height * (0.80 + 0.20 * arch),
                ]
            }
            WalkLegPhase::TerminalSwing => {
                let s = smoothstep01(phase_progress);
                [
                    lerp(overshoot_x, touchdown_x, s),
                    GROUND_Y + lerp(0.55 * step_height, 0.0, s),
                ]
            }
        };

        WalkLegTarget {
            target,
            phase: phase_kind,
            in_stance,
        }
    }

    fn sync_rl_trackers(&mut self) {
        self.rl_episode_time = 0.0;
        self.rl_last_base_x = self.robot_base_x();
        self.rl_last_ball_x = self.ball_x();
        self.rl_last_action_deg = [0.0; 4];
        self.rl_residual_enabled = false;
        self.rl_residual_action_deg = [0.0; 4];
    }

    fn robot_base_x(&self) -> f32 {
        self.robot
            .as_ref()
            .and_then(|robot| self.body_state(robot.torso))
            .map(|body| body.x)
            .unwrap_or(0.0)
    }

    fn ball_x(&self) -> f32 {
        self.ball
            .and_then(|handle| self.body_state(handle))
            .map(|body| body.x)
            .unwrap_or(0.0)
    }

    fn current_relative_targets_deg(&self) -> [f32; 4] {
        let zeros = self.config.servo.zero_offsets;
        let targets = self.current_targets();
        [
            targets
                .right_hip
                .map(|value| (value - zeros.right_hip).to_degrees())
                .unwrap_or_default(),
            targets
                .right_knee
                .map(|value| (value - zeros.right_knee).to_degrees())
                .unwrap_or_default(),
            targets
                .left_hip
                .map(|value| (value - zeros.left_hip).to_degrees())
                .unwrap_or_default(),
            targets
                .left_knee
                .map(|value| (value - zeros.left_knee).to_degrees())
                .unwrap_or_default(),
        ]
    }

    fn rl_done(&self) -> (bool, bool) {
        let observation = self.rl_observation();
        let done = observation.torso_height < self.config.rl.torso_min_height
            || observation.torso_angle.abs() > self.config.rl.torso_max_tilt_rad;
        let truncated = self.rl_episode_time >= self.config.rl.episode_timeout_s;
        (done, truncated)
    }

    fn rl_step_result(&self, forward_progress: f32, ball_progress: f32, action_delta_penalty: f32) -> RlStepResult {
        let observation = self.rl_observation();
        let upright_bonus = observation.torso_angle.cos().max(-1.0);
        let height_span = (self.config.robot.initial_pose.torso.y - self.config.rl.torso_min_height).max(0.1);
        let height_bonus = ((observation.torso_height - self.config.rl.torso_min_height) / height_span).clamp(0.0, 1.0);
        let contact_bonus = if observation.contacts.left_foot || observation.contacts.right_foot {
            1.0
        } else {
            0.0
        };
        let torque_penalty = self
            .state()
            .joints
            .values()
            .map(|joint| joint.torque.abs() / self.servo_max_torque.max(0.0001))
            .sum::<f32>();
        let breakdown = RlRewardBreakdown {
            forward_progress: forward_progress * self.config.rl.reward_forward_weight,
            ball_progress: ball_progress * self.config.rl.reward_ball_forward_weight,
            alive_bonus: self.config.rl.reward_alive_bonus,
            upright_bonus: upright_bonus * self.config.rl.reward_upright_weight,
            height_bonus: height_bonus * self.config.rl.reward_height_weight,
            contact_bonus: contact_bonus * self.config.rl.reward_contact_weight,
            torque_penalty: torque_penalty * self.config.rl.penalty_torque_weight,
            action_delta_penalty: action_delta_penalty * self.config.rl.penalty_action_delta_weight,
        };
        let reward = breakdown.forward_progress
            + breakdown.ball_progress
            + breakdown.alive_bonus
            + breakdown.upright_bonus
            + breakdown.height_bonus
            + breakdown.contact_bonus
            - breakdown.torque_penalty
            - breakdown.action_delta_penalty;
        let (done, truncated) = self.rl_done();
        RlStepResult {
            observation,
            reward,
            done,
            truncated,
            episode_time: self.rl_episode_time,
            breakdown,
        }
    }

    fn body_point(&self, handle: RigidBodyHandle, local: Point<Real>) -> Option<[f32; 2]> {
        self.bodies.get(handle).map(|body| {
            let point = body.position() * local;
            [point.x, point.y]
        })
    }

    fn body_state(&self, handle: RigidBodyHandle) -> Option<BodyState> {
        self.bodies.get(handle).map(|body| BodyState {
            x: body.translation().x,
            y: body.translation().y,
            angle: body.rotation().angle(),
            vx: body.linvel().x,
            vy: body.linvel().y,
            omega: body.angvel(),
        })
    }

    fn set_body_pose(&mut self, handle: RigidBodyHandle, pose: BodyPoseConfig, shift_x: f32) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.set_translation(vector![pose.x + shift_x, pose.y], true);
            body.set_rotation(Rotation::new(pose.angle), true);
            body.set_linvel(vector![0.0, 0.0], true);
            body.set_angvel(0.0, true);
        }
    }

    fn relative_angle(&self, parent: RigidBodyHandle, child: RigidBodyHandle) -> Option<f32> {
        let parent = self.bodies.get(parent)?;
        let child = self.bodies.get(child)?;
        Some(normalize_angle(child.rotation().angle() - parent.rotation().angle()))
    }

    fn relative_velocity(&self, parent: RigidBodyHandle, child: RigidBodyHandle) -> Option<f32> {
        let parent = self.bodies.get(parent)?;
        let child = self.bodies.get(child)?;
        Some(child.angvel() - parent.angvel())
    }

    fn foot_contact(&self, left: bool) -> bool {
        let Some(robot) = &self.robot else {
            return false;
        };
        let collider = if left {
            robot.left_shin_collider
        } else {
            robot.right_shin_collider
        };
        self.narrow_phase
            .contact_pairs_with(collider)
            .any(|pair| pair.has_any_active_contact)
    }

    fn current_targets(&self) -> ServoTargets {
        let mut targets = ServoTargets::default();
        if let Some(robot) = &self.robot {
            for (joint, servo) in &robot.servos {
                match joint {
                    JointName::RightHip => targets.right_hip = Some(servo.target),
                    JointName::RightKnee => targets.right_knee = Some(servo.target),
                    JointName::LeftHip => targets.left_hip = Some(servo.target),
                    JointName::LeftKnee => targets.left_knee = Some(servo.target),
                }
            }
        }
        targets
    }

    fn current_targets_map(&self) -> BTreeMap<String, f32> {
        let mut joints = BTreeMap::new();
        let targets = self.current_targets();
        if let Some(value) = targets.right_hip {
            joints.insert("right_hip".to_owned(), value);
        }
        if let Some(value) = targets.right_knee {
            joints.insert("right_knee".to_owned(), value);
        }
        if let Some(value) = targets.left_hip {
            joints.insert("left_hip".to_owned(), value);
        }
        if let Some(value) = targets.left_knee {
            joints.insert("left_knee".to_owned(), value);
        }
        joints
    }

    fn current_commanded_targets(&self) -> ServoTargets {
        let mut targets = ServoTargets::default();
        if let Some(robot) = &self.robot {
            for (joint, servo) in &robot.servos {
                match joint {
                    JointName::RightHip => targets.right_hip = Some(servo.commanded_target),
                    JointName::RightKnee => targets.right_knee = Some(servo.commanded_target),
                    JointName::LeftHip => targets.left_hip = Some(servo.commanded_target),
                    JointName::LeftKnee => targets.left_knee = Some(servo.commanded_target),
                }
            }
        }
        targets
    }

    fn sync_runtime_settings_into_config(&mut self) {
        let targets = self.current_targets();
        self.config.servo.kp = self.servo_kp;
        self.config.servo.ki = self.servo_ki;
        self.config.servo.kd = self.servo_kd;
        self.config.servo.max_torque = self.servo_max_torque;
        self.config.servo.max_speed_deg_s = self.servo_max_speed_rad_s.to_degrees();
        self.config.walk.nominal_speed_mps = self.walk_target_speed;
        if let Some(robot) = &self.robot {
            if let Some(body) = self.body_state(robot.torso) {
                self.config.robot.initial_pose.torso = BodyPoseConfig {
                    x: body.x,
                    y: body.y,
                    angle: body.angle,
                };
            }
            if let Some(body) = self.body_state(robot.left_thigh) {
                self.config.robot.initial_pose.left_thigh = BodyPoseConfig {
                    x: body.x,
                    y: body.y,
                    angle: body.angle,
                };
            }
            if let Some(body) = self.body_state(robot.left_shin) {
                self.config.robot.initial_pose.left_shin = BodyPoseConfig {
                    x: body.x,
                    y: body.y,
                    angle: body.angle,
                };
            }
            if let Some(body) = self.body_state(robot.right_thigh) {
                self.config.robot.initial_pose.right_thigh = BodyPoseConfig {
                    x: body.x,
                    y: body.y,
                    angle: body.angle,
                };
            }
            if let Some(body) = self.body_state(robot.right_shin) {
                self.config.robot.initial_pose.right_shin = BodyPoseConfig {
                    x: body.x,
                    y: body.y,
                    angle: body.angle,
                };
            }
        }
        self.config.servo.initial_targets = JointAnglesConfig {
            right_hip: targets
                .right_hip
                .unwrap_or(self.config.servo.initial_targets.right_hip),
            right_knee: targets
                .right_knee
                .unwrap_or(self.config.servo.initial_targets.right_knee),
            left_hip: targets
                .left_hip
                .unwrap_or(self.config.servo.initial_targets.left_hip),
            left_knee: targets
                .left_knee
                .unwrap_or(self.config.servo.initial_targets.left_knee),
        };
        self.canonical_initial_pose = self.config.robot.initial_pose.clone();
        self.canonical_zero_offsets = self.config.servo.zero_offsets;
        self.canonical_initial_targets = self.config.servo.initial_targets;
    }

    fn dt(&self) -> f32 {
        self.integration_parameters.dt
    }
}

fn mirror_body_pose(pose: BodyPoseConfig) -> BodyPoseConfig {
    BodyPoseConfig {
        x: -pose.x,
        y: pose.y,
        angle: -pose.angle,
    }
}

fn mirror_initial_pose(pose: &InitialPoseConfig) -> InitialPoseConfig {
    InitialPoseConfig {
        torso: mirror_body_pose(pose.torso),
        left_thigh: mirror_body_pose(pose.left_thigh),
        left_shin: mirror_body_pose(pose.left_shin),
        right_thigh: mirror_body_pose(pose.right_thigh),
        right_shin: mirror_body_pose(pose.right_shin),
    }
}

fn mirror_joint_angles(angles: JointAnglesConfig) -> JointAnglesConfig {
    JointAnglesConfig {
        right_hip: -angles.right_hip,
        right_knee: -angles.right_knee,
        left_hip: -angles.left_hip,
        left_knee: -angles.left_knee,
    }
}

impl JointServo {
    fn new(parent: RigidBodyHandle, child: RigidBodyHandle, target: f32) -> Self {
        Self {
            parent,
            child,
            target,
            commanded_target: target,
            last_torque: 0.0,
            integral_error: 0.0,
        }
    }
}

fn interpolate_joint_map(
    start: &BTreeMap<String, f32>,
    end: &BTreeMap<String, f32>,
    alpha: f32,
) -> BTreeMap<String, f32> {
    let mut joints = BTreeMap::new();
    for name in ["right_hip", "right_knee", "left_hip", "left_knee"] {
        let start_value = start.get(name).copied().unwrap_or_default();
        let end_value = end.get(name).copied().unwrap_or(start_value);
        joints.insert(name.to_owned(), start_value + (end_value - start_value) * alpha);
    }
    joints
}

fn enforce_min_foot_separation(left_target: &mut [f32; 2], right_target: &mut [f32; 2], min_separation: f32) {
    let min_separation = min_separation.max(0.0);
    let dx = right_target[0] - left_target[0];
    if dx.abs() >= min_separation {
        return;
    }
    let midpoint = 0.5 * (left_target[0] + right_target[0]);
    let sign = if dx >= 0.0 { 1.0 } else { -1.0 };
    let half = 0.5 * min_separation;
    left_target[0] = midpoint - sign * half;
    right_target[0] = midpoint + sign * half;
}

fn classify_walk_leg_phase(cycle: f32, stance_duty: f32) -> (WalkLegPhase, f32, bool) {
    let cycle = cycle.rem_euclid(1.0);
    let stance_duty = stance_duty.clamp(0.50, 0.85);
    if cycle < stance_duty {
        let loading_end = stance_duty * 0.18;
        let mid_end = stance_duty * 0.74;
        if cycle < loading_end {
            (
                WalkLegPhase::LoadingResponse,
                (cycle / loading_end.max(1e-4)).clamp(0.0, 1.0),
                true,
            )
        } else if cycle < mid_end {
            (
                WalkLegPhase::MidStance,
                ((cycle - loading_end) / (mid_end - loading_end).max(1e-4)).clamp(0.0, 1.0),
                true,
            )
        } else {
            (
                WalkLegPhase::PreSwing,
                ((cycle - mid_end) / (stance_duty - mid_end).max(1e-4)).clamp(0.0, 1.0),
                true,
            )
        }
    } else {
        let swing_phase = ((cycle - stance_duty) / (1.0 - stance_duty)).clamp(0.0, 1.0);
        if swing_phase < 0.28 {
            (
                WalkLegPhase::InitialSwing,
                (swing_phase / 0.28).clamp(0.0, 1.0),
                false,
            )
        } else if swing_phase < 0.72 {
            (
                WalkLegPhase::MidSwing,
                ((swing_phase - 0.28) / 0.44).clamp(0.0, 1.0),
                false,
            )
        } else {
            (
                WalkLegPhase::TerminalSwing,
                ((swing_phase - 0.72) / 0.28).clamp(0.0, 1.0),
                false,
            )
        }
    }
}

fn smoothstep01(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn opposite_leg(side: LegSide) -> LegSide {
    match side {
        LegSide::Left => LegSide::Right,
        LegSide::Right => LegSide::Left,
    }
}

fn normalize_angle(angle: f32) -> f32 {
    let mut wrapped = angle;
    while wrapped > std::f32::consts::PI {
        wrapped -= std::f32::consts::TAU;
    }
    while wrapped < -std::f32::consts::PI {
        wrapped += std::f32::consts::TAU;
    }
    wrapped
}

fn clamp_joint_target(_joint: JointName, angle: f32) -> f32 {
    angle.clamp(-std::f32::consts::PI, std::f32::consts::PI)
}
