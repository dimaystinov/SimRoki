mod config;

use nalgebra::{point, vector};
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

pub use config::{
    BodyDynamicsConfig, BodyPoseConfig, InitialPoseConfig, JointAnglesConfig, LinkConfig, PhysicsConfig, RobotConfig,
    ServoConfig, SimulationConfig,
};

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
const ROBOT_BALL_OFFSET_X_M: f32 = 0.7;
const ROBOT_BALL_CLEARANCE_Y_M: f32 = 0.02;

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
        sim.spawn_robot_ball();
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
                .angular_damping(0.2)
                .linear_damping(0.05)
                .build(),
        );
        let collider = ColliderBuilder::ball(radius)
            .collision_groups(InteractionGroups::new(
                GROUP_BALL,
                all_robot_groups() | GROUP_GROUND | GROUP_BALL | GROUP_ENVIRONMENT,
            ))
            .density(density)
            .friction(0.85)
            .restitution(0.1)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        self.ball = Some(handle);
    }

    fn spawn_robot_ball(&mut self) {
        let pose = &self.config.robot.initial_pose;
        let rightmost_x = pose
            .torso
            .x
            .max(pose.left_thigh.x)
            .max(pose.left_shin.x)
            .max(pose.right_thigh.x)
            .max(pose.right_shin.x);
        let ball_x = rightmost_x + ROBOT_BALL_OFFSET_X_M;
        let ball_y = ROBOT_BALL_RADIUS_M + ROBOT_BALL_CLEARANCE_Y_M;
        self.spawn_ball(
            vector![ball_x, ball_y],
            ROBOT_BALL_RADIUS_M,
            ROBOT_BALL_MASS_KG,
        );
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
        let targets = self.current_targets();
        let gains = self.servo_gains();
        *self = Self::new_robot_with_config_and_suspension(self.config.clone(), self.robot_suspended);
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        self.apply_targets(targets);
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
        *self = match scene {
            SceneKind::Ball => Self::new_ball_with_config(self.config.clone()),
            SceneKind::Robot => Self::new_robot_with_config_and_suspension(self.config.clone(), self.robot_suspended),
        };
    }

    pub fn set_robot_suspended(&mut self, suspended: bool) {
        let targets = self.current_targets();
        let gains = self.servo_gains();
        *self = Self::new_robot_with_config_and_suspension(self.config.clone(), suspended);
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        self.apply_targets(targets);
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
            *self = Self::new_robot_with_config_and_suspension(config, suspended);
            self.paused = paused;
        }
    }

    pub fn set_servo_gains(&mut self, kp: f32, ki: f32, kd: f32, max_torque: f32) {
        self.servo_kp = kp.clamp(-20.0, 20.0);
        self.servo_ki = ki.clamp(-5.0, 5.0);
        self.servo_kd = kd.clamp(-1.0, 1.0);
        self.servo_max_torque = max_torque.clamp(0.5, 100.0);
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
                servo.target = clamp_joint_target(joint, angle);
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
            self.apply_motion_sequence_targets();
            self.apply_gait_targets();
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
        }
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

    fn sync_runtime_settings_into_config(&mut self) {
        let targets = self.current_targets();
        self.config.servo.kp = self.servo_kp;
        self.config.servo.ki = self.servo_ki;
        self.config.servo.kd = self.servo_kd;
        self.config.servo.max_torque = self.servo_max_torque;
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
    }

    fn dt(&self) -> f32 {
        self.integration_parameters.dt
    }
}

impl JointServo {
    fn new(parent: RigidBodyHandle, child: RigidBodyHandle, target: f32) -> Self {
        Self {
            parent,
            child,
            target,
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
