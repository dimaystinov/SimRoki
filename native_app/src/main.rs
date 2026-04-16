use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};
use serde::{Deserialize, Serialize};
use sim_core::{
    GaitCommand, JointCommand, MotionSequenceCommand, PoseCommand, RlObservation, RlStepCommand,
    RlStepResult, RlResetCommand, SceneKind, ServoTargets, Simulation, SimulationConfig,
    SimulationState, WalkConfigCommand, WalkDirectionCommand,
};
use std::{
    fs,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::runtime::Builder;
use tracing::{error, info, warn};

const WINDOW_WIDTH: i32 = 1280;
const WINDOW_HEIGHT: i32 = 800;
const PIXELS_PER_METER: f32 = 280.0;
const ZOOM_MIN: f32 = 0.1;
const ZOOM_MAX: f32 = 2.8;
const ZOOM_SMOOTHING_HZ: f32 = 10.0;
const ZOOM_WHEEL_FACTOR: f32 = 1.12;
const GRID_MINOR_STEP_WORLD: f32 = 0.5;
const GRID_MAJOR_EVERY: i32 = 4;
const GROUND_Y: f32 = 0.0;
const PANEL_WIDTH: f32 = 440.0;
const PANEL_MIN_WIDTH: f32 = 320.0;
const PANEL_MARGIN: f32 = 12.0;
const CONFIG_PATH: &str = "robot_config.toml";
const MOTION_JSON_PATH: &str = "motion_sequence.json";

type SharedSimulation = Arc<Mutex<Simulation>>;
type SharedExternalControl = Arc<Mutex<ExternalControlState>>;

struct AppFonts {
    cyrillic: Option<Font>,
}

#[derive(Clone)]
struct AppState {
    sim: SharedSimulation,
    external_control: SharedExternalControl,
}

#[derive(Debug, Deserialize)]
struct SceneRequest {
    scene: SceneKind,
}

#[derive(Debug, Serialize)]
struct OkResponse {
    ok: bool,
}

#[derive(Debug, Clone)]
struct ViewState {
    zoom: f32,
    zoom_target: f32,
    focus: Vec2,
    follow_robot: bool,
    is_panning: bool,
    last_mouse: Vec2,
}

#[derive(Debug, Clone)]
struct ControlPanelState {
    right_hip: f32,
    right_knee: f32,
    left_hip: f32,
    left_knee: f32,
    right_hip_zero: f32,
    right_knee_zero: f32,
    left_hip_zero: f32,
    left_knee_zero: f32,
    servo_kp: f32,
    servo_ki: f32,
    servo_kd: f32,
    servo_max_torque: f32,
    suspend_height: f32,
    motion_frame_ms: f32,
    motion_frames: Vec<[f32; 5]>,
    motion_json: String,
    motion_loop: bool,
    motion_repeat_delay_ms: f32,
    robofest_active: bool,
    robofest_start_time: f32,
    robofest_elapsed_time: f32,
    robofest_finish_x: Option<f32>,
    robofest_robot_reached: bool,
    robofest_ball_reached: bool,
    robofest_result_message: Option<String>,
    initialized: bool,
}

#[derive(Debug)]
struct ExternalControlState {
    last_external_command: Option<Instant>,
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn lock() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "simulation lock poisoned".to_owned(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (self.status, self.message).into_response()
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Five Link Robot Simulator".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        high_dpi: true,
        sample_count: 4,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .try_init();

    let sim = Arc::new(Mutex::new(load_simulation()));
    let external_control = Arc::new(Mutex::new(ExternalControlState::default()));
    let fonts = AppFonts {
        cyrillic: load_ttf_font("C:/Windows/Fonts/arial.ttf").await.ok(),
    };
    let mut view = ViewState {
        zoom: 1.0,
        zoom_target: 1.0,
        focus: vec2(0.0, 0.85),
        follow_robot: true,
        is_panning: false,
        last_mouse: vec2(0.0, 0.0),
    };
    let mut controls = ControlPanelState::default();
    spawn_api_server(sim.clone(), external_control.clone());

    loop {
        clear_background(color_u8!(244, 241, 234, 255));
        handle_keyboard(&sim);
        update_view(&mut view, &sim);
        update_robofest_state(&sim, &mut controls);
        draw_world(&sim, &view, &controls, &fonts);
        draw_control_panel(&sim, &external_control, &mut controls);
        let allow_frame_stepping = !external_control_active(&external_control);
        if allow_frame_stepping {
            if let Ok(mut sim) = sim.lock() {
                sim.step_for_seconds(get_frame_time());
            }
        } else if let Ok(sim) = sim.lock() {
            let _ = sim.state();
        }
        next_frame().await;
    }
}

fn load_simulation() -> Simulation {
    match Simulation::from_config_file(CONFIG_PATH) {
        Ok(sim) => {
            info!("loaded config from {CONFIG_PATH}");
            sim
        }
        Err(err) => {
            warn!("config load failed, writing default config to {CONFIG_PATH}: {err}");
            let config = SimulationConfig::default();
            if let Err(save_err) = config.save_to_file(CONFIG_PATH) {
                error!("failed to write default config: {save_err}");
            }
            Simulation::from_config(config)
        }
    }
}

fn spawn_api_server(sim: SharedSimulation, external_control: SharedExternalControl) {
    thread::spawn(move || {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        runtime.block_on(async move {
            if let Err(err) = run_server(sim, external_control).await {
                error!("control API stopped: {err:?}");
            }
        });
    });
}

async fn run_server(sim: SharedSimulation, external_control: SharedExternalControl) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health))
        .route("/state", get(get_state))
        .route("/pause", post(pause))
        .route("/resume", post(resume))
        .route("/reset", post(reset_robot))
        .route("/reset/ball", post(reset_ball))
        .route("/config/save", post(save_config))
        .route("/scene", post(set_scene))
        .route("/joint/angle", post(set_joint))
        .route("/servo/targets", post(set_targets))
        .route("/pose", post(set_pose))
        .route("/gait", post(set_gait))
        .route("/motion/sequence_deg", post(set_motion_sequence_deg))
        .route("/rl/reset", post(rl_reset))
        .route("/rl/observation", get(rl_observation))
        .route("/rl/step", post(rl_step))
        .route("/walk/direction", post(set_walk_direction))
        .route("/walk/config", post(set_walk_config))
        .with_state(AppState { sim, external_control });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    info!("native viewer active; control API at http://127.0.0.1:8080");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<OkResponse> {
    Json(OkResponse { ok: true })
}

async fn get_state(State(state): State<AppState>) -> Result<Json<SimulationState>, AppError> {
    let sim = state.sim.lock().map_err(|_| AppError::lock())?;
    Ok(Json(sim.state()))
}

async fn pause(State(state): State<AppState>) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.pause();
    Ok(Json(OkResponse { ok: true }))
}

async fn resume(State(state): State<AppState>) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.resume();
    Ok(Json(OkResponse { ok: true }))
}

async fn reset_robot(State(state): State<AppState>) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.reset_robot();
    Ok(Json(OkResponse { ok: true }))
}

async fn reset_ball(State(state): State<AppState>) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.reset_ball();
    Ok(Json(OkResponse { ok: true }))
}

async fn save_config(State(state): State<AppState>) -> Result<Json<OkResponse>, AppError> {
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.save_config_file(CONFIG_PATH)
        .map_err(|message| AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message,
        })?;
    Ok(Json(OkResponse { ok: true }))
}

async fn set_scene(
    State(state): State<AppState>,
    Json(request): Json<SceneRequest>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.set_scene(request.scene);
    Ok(Json(OkResponse { ok: true }))
}

async fn set_joint(
    State(state): State<AppState>,
    Json(command): Json<JointCommand>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.set_joint_target(command.joint, command.angle);
    sim.clear_gait();
    Ok(Json(OkResponse { ok: true }))
}

async fn set_targets(
    State(state): State<AppState>,
    Json(targets): Json<ServoTargets>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.apply_targets(targets);
    sim.clear_gait();
    Ok(Json(OkResponse { ok: true }))
}

async fn set_pose(
    State(state): State<AppState>,
    Json(pose): Json<PoseCommand>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.apply_pose(pose);
    sim.clear_gait();
    Ok(Json(OkResponse { ok: true }))
}

async fn set_gait(
    State(state): State<AppState>,
    Json(gait): Json<GaitCommand>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.set_gait(gait);
    Ok(Json(OkResponse { ok: true }))
}

async fn rl_reset(
    State(state): State<AppState>,
    payload: Option<Json<RlResetCommand>>,
) -> Result<Json<RlStepResult>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    let command = payload.map(|value| value.0).unwrap_or_default();
    Ok(Json(sim.rl_reset_with(command)))
}

async fn rl_observation(State(state): State<AppState>) -> Result<Json<RlObservation>, AppError> {
    let sim = state.sim.lock().map_err(|_| AppError::lock())?;
    Ok(Json(sim.rl_observation()))
}

async fn rl_step(
    State(state): State<AppState>,
    Json(command): Json<RlStepCommand>,
) -> Result<Json<RlStepResult>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    Ok(Json(sim.rl_step_deg(command)))
}

async fn set_walk_direction(
    State(state): State<AppState>,
    Json(command): Json<WalkDirectionCommand>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.set_target_direction(command.direction);
    if let Some(speed_mps) = command.speed_mps {
        sim.set_walk_target_speed(speed_mps);
    }
    if let Some(enabled) = command.enabled {
        sim.set_directional_walk_enabled(enabled);
    }
    Ok(Json(OkResponse { ok: true }))
}

async fn set_walk_config(
    State(state): State<AppState>,
    Json(command): Json<WalkConfigCommand>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.update_walk_config(command);
    Ok(Json(OkResponse { ok: true }))
}

async fn set_motion_sequence_deg(
    State(state): State<AppState>,
    Json(sequence): Json<Vec<[f32; 5]>>,
) -> Result<Json<OkResponse>, AppError> {
    mark_external_control(&state.external_control);
    let mut sim = state.sim.lock().map_err(|_| AppError::lock())?;
    sim.set_motion_sequence_deg(MotionSequenceCommand {
        frames: sequence,
        loop_enabled: false,
        repeat_delay_ms: 0.0,
    });
    Ok(Json(OkResponse { ok: true }))
}

fn handle_keyboard(sim: &SharedSimulation) {
    if is_key_pressed(KeyCode::R) {
        if let Ok(mut sim) = sim.lock() {
            sim.reset_robot();
        }
    }
    if is_key_pressed(KeyCode::B) {
        if let Ok(mut sim) = sim.lock() {
            sim.reset_ball();
        }
    }
    if is_key_pressed(KeyCode::Left) {
        if let Ok(mut sim) = sim.lock() {
            sim.set_target_direction(-1.0);
            sim.set_directional_walk_enabled(true);
        }
    }
    if is_key_pressed(KeyCode::Right) {
        if let Ok(mut sim) = sim.lock() {
            sim.set_target_direction(1.0);
            sim.set_directional_walk_enabled(true);
        }
    }
    if is_key_pressed(KeyCode::S) {
        if let Ok(mut sim) = sim.lock() {
            sim.set_directional_walk_enabled(false);
        }
    }
}

fn update_view(view: &mut ViewState, sim: &SharedSimulation) {
    let mouse = vec2(mouse_position().0, mouse_position().1);
    let pan_pressed = is_mouse_button_down(MouseButton::Middle)
        || (is_mouse_button_down(MouseButton::Left) && is_key_down(KeyCode::LeftShift))
        || (is_mouse_button_down(MouseButton::Right) && !is_key_down(KeyCode::LeftControl));
    if pan_pressed {
        if view.is_panning {
            let delta = mouse - view.last_mouse;
            view.focus.x -= delta.x / (PIXELS_PER_METER * view.zoom);
            view.focus.y += delta.y / (PIXELS_PER_METER * view.zoom);
        }
        view.is_panning = true;
        view.follow_robot = false;
    } else {
        view.is_panning = false;
    }
    view.last_mouse = mouse;

    if is_key_pressed(KeyCode::F) || is_key_pressed(KeyCode::Space) {
        view.follow_robot = true;
    }

    let raw_wheel = mouse_wheel().1;
    let wheel = normalize_wheel_delta(raw_wheel);
    if wheel.abs() > f32::EPSILON {
        view.zoom_target = (view.zoom_target * ZOOM_WHEEL_FACTOR.powf(wheel)).clamp(ZOOM_MIN, ZOOM_MAX);
    }
    let smoothing = 1.0 - (-ZOOM_SMOOTHING_HZ * get_frame_time()).exp();
    view.zoom += (view.zoom_target - view.zoom) * smoothing;

    let state = {
        let Ok(sim) = sim.lock() else {
            return;
        };
        sim.state()
    };
    if view.follow_robot {
        if let Some(base) = state.base {
            view.focus = vec2(base.x, base.y * 0.75);
        } else if let Some(ball) = state.ball {
            view.focus = vec2(ball.x, ball.y);
        }
    } else if let Some(ball) = state.ball {
        if state.scene == SceneKind::Ball {
            view.focus = vec2(ball.x, ball.y);
        }
    }
}

fn normalize_wheel_delta(raw_wheel: f32) -> f32 {
    if raw_wheel.abs() > 10.0 {
        raw_wheel / 120.0
    } else {
        raw_wheel
    }
}

fn draw_world(
    sim: &SharedSimulation,
    view: &ViewState,
    controls: &ControlPanelState,
    fonts: &AppFonts,
) {
    let state = {
        let Ok(sim) = sim.lock() else {
            return;
        };
        sim.state()
    };

    draw_grid(view);
    if let Some(finish_x) = controls.robofest_finish_x {
        draw_finish_line(finish_x, view);
    }
    match state.scene {
        SceneKind::Ball => draw_ball_scene(&state, view),
        SceneKind::Robot => draw_robot_scene(sim, &state, view),
    }
    draw_overlay(&state, view, controls, fonts);
}

fn draw_control_panel(
    sim: &SharedSimulation,
    external_control: &SharedExternalControl,
    controls: &mut ControlPanelState,
) {
    let state = {
        let Ok(sim) = sim.lock() else {
            return;
        };
        sim.state()
    };
    if let Ok(sim) = sim.lock() {
        let (kp, ki, kd, max_torque) = sim.servo_gains();
        controls.servo_kp = kp;
        controls.servo_ki = ki;
        controls.servo_kd = kd;
        controls.servo_max_torque = max_torque;
        controls.suspend_height = sim.suspend_clearance();
    }
    let external_active = external_control_active(external_control);
    if !controls.initialized {
        controls.sync_from_state(&state);
    }

    let mut changed = false;
    let mut suspend_changed = false;
    let available_width = (screen_width() - PANEL_MARGIN * 2.0).max(PANEL_MIN_WIDTH);
    let panel_width = PANEL_WIDTH.min(available_width).max(PANEL_MIN_WIDTH.min(available_width));
    let panel_height = (screen_height() - PANEL_MARGIN * 2.0).max(220.0);
    let panel_x = (screen_width() - panel_width - PANEL_MARGIN).max(PANEL_MARGIN);
    let panel_y = PANEL_MARGIN.min((screen_height() - panel_height).max(0.0));
    widgets::Window::new(
        hash!("servo-panel"),
        vec2(panel_x, panel_y),
        vec2(panel_width, panel_height),
    )
    .label("Servo Control")
    .titlebar(true)
    .movable(false)
    .ui(&mut *root_ui(), |ui| {
        ui.label(None, &format!("time {:.2}s | {}", state.time, state.mode));
        ui.label(
            None,
            if external_active {
                "control source: external API"
            } else {
                "control source: built-in sliders"
            },
        );
        ui.label(None, "servo map: [1] right_hip  [2] right_knee  [3] left_hip  [4] left_knee");
        ui.separator();
        changed |= slider_row(ui, "[1] right_hip", &mut controls.right_hip, -std::f32::consts::PI..std::f32::consts::PI);
        joint_status_row(ui, &state, "right_hip", controls.right_hip_zero);
        changed |= slider_row(ui, "[2] right_knee", &mut controls.right_knee, -std::f32::consts::PI..std::f32::consts::PI);
        joint_status_row(ui, &state, "right_knee", controls.right_knee_zero);
        changed |= slider_row(ui, "[3] left_hip", &mut controls.left_hip, -std::f32::consts::PI..std::f32::consts::PI);
        joint_status_row(ui, &state, "left_hip", controls.left_hip_zero);
        changed |= slider_row(ui, "[4] left_knee", &mut controls.left_knee, -std::f32::consts::PI..std::f32::consts::PI);
        joint_status_row(ui, &state, "left_knee", controls.left_knee_zero);
        ui.separator();
        ui.label(None, "PID gains");
        changed |= slider_row(ui, "kp", &mut controls.servo_kp, -20.0..20.0);
        changed |= slider_row(ui, "ki", &mut controls.servo_ki, -5.0..5.0);
        changed |= slider_row(ui, "kd", &mut controls.servo_kd, -1.0..1.0);
        changed |= slider_row(ui, "max_torque", &mut controls.servo_max_torque, 0.1..10.0);
        ui.separator();
        ui.label(None, "Suspend point");
        suspend_changed = slider_row(ui, "suspend_height", &mut controls.suspend_height, 0.05..1.8);
        changed |= suspend_changed;
        if changed {
            if let Ok(mut sim) = sim.lock() {
                sim.set_servo_gains(
                    controls.servo_kp,
                    controls.servo_ki,
                    controls.servo_kd,
                    controls.servo_max_torque,
                );
                if suspend_changed {
                    sim.set_suspend_clearance(controls.suspend_height);
                }
            }
        }
        ui.separator();

        if ui.button(None, "Apply sliders") {
            apply_controls(sim, controls);
        }
        if ui.button(None, "Add frame from sliders") {
            controls.motion_frames.push(current_motion_frame(controls));
            sync_motion_json_from_frames(controls);
        }
        if ui.button(None, "Replace last frame") {
            let frame = current_motion_frame(controls);
            if let Some(last) = controls.motion_frames.last_mut() {
                *last = frame;
                sync_motion_json_from_frames(controls);
            }
        }
        if ui.button(None, "Remove last frame") {
            let _ = controls.motion_frames.pop();
            sync_motion_json_from_frames(controls);
        }
        if ui.button(None, "Clear motion") {
            controls.motion_frames.clear();
            sync_motion_json_from_frames(controls);
        }
        if ui.button(None, "Play motion") {
            if apply_motion_json_to_frames(controls).is_ok() && !controls.motion_frames.is_empty() {
                if let Ok(mut sim) = sim.lock() {
                    sim.clear_gait();
                    sim.set_motion_sequence_deg(MotionSequenceCommand {
                        frames: controls.motion_frames.clone(),
                        loop_enabled: controls.motion_loop,
                        repeat_delay_ms: controls.motion_repeat_delay_ms,
                    });
                }
            }
        }
        if ui.button(None, "Start ROBOFEST 2026") {
            if apply_motion_json_to_frames(controls).is_ok() && !controls.motion_frames.is_empty() {
                if let Ok(mut sim) = sim.lock() {
                    sim.reset_robot();
                    sim.resume();
                    sim.clear_gait();
                    sim.set_motion_sequence_deg(MotionSequenceCommand {
                        frames: controls.motion_frames.clone(),
                        loop_enabled: false,
                        repeat_delay_ms: 0.0,
                    });
                    let fresh_state = sim.state();
                    let start_x = fresh_state.base.as_ref().map(|base| base.x).unwrap_or(0.0);
                    controls.robofest_active = true;
                    controls.robofest_start_time = fresh_state.time;
                    controls.robofest_elapsed_time = 0.0;
                    controls.robofest_finish_x = Some(start_x + 100.0);
                    controls.robofest_robot_reached = false;
                    controls.robofest_ball_reached = false;
                    controls.robofest_result_message = None;
                    controls.sync_from_state(&fresh_state);
                }
            }
        }
        if ui.button(None, "Save JSON") {
            let _ = apply_motion_json_to_frames(controls);
            if let Err(err) = fs::write(MOTION_JSON_PATH, &controls.motion_json) {
                error!("failed to save motion json: {err}");
            } else {
                info!("saved motion json to {MOTION_JSON_PATH}");
            }
        }
        if ui.button(None, "Load JSON") {
            match fs::read_to_string(MOTION_JSON_PATH) {
                Ok(text) => {
                    controls.motion_json = text;
                    if let Err(err) = apply_motion_json_to_frames(controls) {
                        error!("failed to parse loaded motion json: {err}");
                    }
                }
                Err(err) => error!("failed to load motion json: {err}"),
            }
        }
        if ui.button(None, "Use current pose as zero") {
            if let Ok(mut sim) = sim.lock() {
                sim.set_servo_zero_to_current_pose();
                controls.sync_from_state(&sim.state());
            }
            apply_controls(sim, controls);
        }
        if ui.button(None, "Save config") {
            if let Ok(mut sim) = sim.lock() {
                if let Err(err) = sim.save_config_file(CONFIG_PATH) {
                    error!("failed to save config: {err}");
                } else {
                    info!("saved config to {CONFIG_PATH}");
                }
            }
        }
        if ui.button(
            None,
            if state.base.is_some() {
                if sim.lock().ok().map(|sim| sim.robot_suspended()).unwrap_or(false) {
                    "Unsuspend top point"
                } else {
                    "Suspend top point"
                }
            } else {
                "Suspend top point"
            },
        ) {
            if let Ok(mut sim) = sim.lock() {
                let next = !sim.robot_suspended();
                sim.set_robot_suspended(next);
                controls.sync_from_state(&sim.state());
            }
        }
        if ui.button(None, "Reset robot") {
            if let Ok(mut sim) = sim.lock() {
                sim.reset_robot();
                controls.sync_from_state(&sim.state());
            }
        }
        if ui.button(None, if state.paused { "Resume" } else { "Pause" }) {
            if let Ok(mut sim) = sim.lock() {
                if state.paused {
                    sim.resume();
                } else {
                    sim.pause();
                }
            }
        }
        if ui.button(None, "Sync from robot") {
            controls.sync_from_state(&state);
        }
        if ui.button(None, "Walk left") {
            if let Ok(mut sim) = sim.lock() {
                sim.set_target_direction(-1.0);
                sim.set_directional_walk_enabled(true);
            }
        }
        if ui.button(None, "Walk right") {
            if let Ok(mut sim) = sim.lock() {
                sim.set_target_direction(1.0);
                sim.set_directional_walk_enabled(true);
            }
        }
        if ui.button(None, "Stop walk") {
            if let Ok(mut sim) = sim.lock() {
                sim.set_directional_walk_enabled(false);
            }
        }

        ui.separator();
        ui.label(None, "Debug info");
        ui.label(
            None,
            &format!(
                "walk: {} | dir: {} | speed: {:.2}/{:.2} m/s",
                state.walk_enabled,
                if state.walk_direction >= 0.0 { "right" } else { "left" },
                state.walk_speed,
                state.walk_target_speed
            ),
        );
        ui.label(None, "Motion editor");
        let frame_changed = slider_row(ui, "frame_ms", &mut controls.motion_frame_ms, 10.0..1000.0);
        if frame_changed {
            controls.motion_frame_ms = (controls.motion_frame_ms / 10.0).round() * 10.0;
            controls.motion_frame_ms = controls.motion_frame_ms.clamp(10.0, 1000.0);
        }
        ui.label(None, "Loop motion");
        widgets::Checkbox::new(hash!("motion_loop")).ui(ui, &mut controls.motion_loop);
        let repeat_changed = slider_row(
            ui,
            "repeat_delay_ms",
            &mut controls.motion_repeat_delay_ms,
            10.0..1000.0,
        );
        if repeat_changed {
            controls.motion_repeat_delay_ms = (controls.motion_repeat_delay_ms / 10.0).round() * 10.0;
            controls.motion_repeat_delay_ms = controls.motion_repeat_delay_ms.clamp(10.0, 1000.0);
        }
        ui.label(
            None,
            "frame format: [time_ms, servo1, servo2, servo3, servo4] in degrees",
        );
        ui.editbox(
            hash!("motion_json_editbox"),
            vec2((panel_width - 36.0).max(180.0), 180.0),
            &mut controls.motion_json,
        );
        ui.separator();
        ui.label(None, &format!("L foot: {}", state.contacts.left_foot));
        ui.label(None, &format!("R foot: {}", state.contacts.right_foot));
        ui.label(
            None,
            &format!(
                "suspended: {}",
                sim.lock().ok().map(|sim| sim.robot_suspended()).unwrap_or(false)
            ),
        );
        ui.separator();
        ui.label(None, "Link masses");
        for name in ["torso", "left_thigh", "left_shin", "right_thigh", "right_shin"] {
            if let Some(mass) = state.link_masses.get(name) {
                ui.label(None, &format!("{name}: {:.2} kg", mass));
            }
        }
        ui.separator();
        ui.label(None, "Link lengths");
        for name in ["torso", "left_thigh", "left_shin", "right_thigh", "right_shin"] {
            if let Some(length) = state.link_lengths.get(name) {
                ui.label(None, &format!("{name}: {:.2} m", length));
            }
        }
    });

    if changed && !external_active {
        apply_controls(sim, controls);
    }
}

fn slider_row(
    ui: &mut macroquad::ui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::Range<f32>,
) -> bool {
    let before = *value;
    let text = if label.contains("hip") || label.contains("knee") {
        format!("{label}: {:.1}°", value.to_degrees())
    } else {
        format!("{label}: {:.2}", *value)
    };
    ui.label(None, &text);
    widgets::Slider::new(hash!(label), range).ui(ui, value);
    (before - *value).abs() > 0.0001
}

fn joint_status_row(
    ui: &mut macroquad::ui::Ui,
    state: &SimulationState,
    joint_name: &str,
    zero: f32,
) {
    if let Some(joint) = state.joints.get(joint_name) {
        ui.label(
            None,
            &format!(
                "current: {:.1}° | target: {:.1}° | zero: {:.1}° | used: {:.1}°",
                (joint.angle - zero).to_degrees(),
                (joint.target - zero).to_degrees(),
                zero.to_degrees(),
                joint.target.to_degrees()
            ),
        );
    }
}

fn apply_controls(sim: &SharedSimulation, controls: &ControlPanelState) {
    if let Ok(mut sim) = sim.lock() {
        sim.clear_gait();
        sim.apply_targets(ServoTargets {
            right_hip: Some(controls.right_hip_zero + controls.right_hip),
            right_knee: Some(controls.right_knee_zero + controls.right_knee),
            left_hip: Some(controls.left_hip_zero + controls.left_hip),
            left_knee: Some(controls.left_knee_zero + controls.left_knee),
        });
    }
}

fn update_robofest_state(sim: &SharedSimulation, controls: &mut ControlPanelState) {
    let state = {
        let Ok(sim) = sim.lock() else {
            return;
        };
        sim.state()
    };

    let Some(finish_x) = controls.robofest_finish_x else {
        return;
    };

    controls.robofest_elapsed_time = (state.time - controls.robofest_start_time).max(0.0);
    controls.robofest_robot_reached = state.base.as_ref().map(|base| base.x >= finish_x).unwrap_or(false);
    controls.robofest_ball_reached = state.ball.as_ref().map(|ball| ball.x >= finish_x).unwrap_or(false);

    if controls.robofest_active && controls.robofest_robot_reached {
        controls.robofest_active = false;
        let ball_status = if controls.robofest_ball_reached {
            "мяч: да (+10 баллов)"
        } else {
            "мяч: нет (+0 баллов)"
        };
        controls.robofest_result_message = Some(format!(
            "Поздравляем! Время: {:.2} с | {ball_status}",
            controls.robofest_elapsed_time
        ));
    }
}

fn current_motion_frame(controls: &ControlPanelState) -> [f32; 5] {
    [
        controls.motion_frame_ms.max(1.0),
        (controls.right_hip_zero + controls.right_hip).to_degrees(),
        (controls.right_knee_zero + controls.right_knee).to_degrees(),
        (controls.left_hip_zero + controls.left_hip).to_degrees(),
        (controls.left_knee_zero + controls.left_knee).to_degrees(),
    ]
}

fn sync_motion_json_from_frames(controls: &mut ControlPanelState) {
    controls.motion_json = format_motion_frames_json(&controls.motion_frames);
}

fn apply_motion_json_to_frames(controls: &mut ControlPanelState) -> Result<(), String> {
    let frames: Vec<[f32; 5]> = serde_json::from_str(&controls.motion_json)
        .map_err(|err| format!("invalid motion json: {err}"))?;
    controls.motion_frames = frames;
    controls.motion_json = format_motion_frames_json(&controls.motion_frames);
    Ok(())
}

fn format_motion_frames_json(frames: &[[f32; 5]]) -> String {
    if frames.is_empty() {
        return "[]".to_owned();
    }

    let lines = frames
        .iter()
        .map(|frame| {
            format!(
                "  [{:.1}, {:.5}, {:.5}, {:.5}, {:.5}]",
                frame[0], frame[1], frame[2], frame[3], frame[4]
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format!("[\n{}\n]", lines)
}

fn draw_grid(view: &ViewState) {
    let half_w = screen_width() * 0.5 / (PIXELS_PER_METER * view.zoom);
    let half_h = screen_height() * 0.5 / (PIXELS_PER_METER * view.zoom);
    let left = view.focus.x - half_w;
    let right = view.focus.x + half_w;
    let bottom = view.focus.y - half_h;
    let top = view.focus.y + half_h;

    let start_x = (left / GRID_MINOR_STEP_WORLD).floor() as i32 - 1;
    let end_x = (right / GRID_MINOR_STEP_WORLD).ceil() as i32 + 1;
    for x in start_x..=end_x {
        let world_x = x as f32 * GRID_MINOR_STEP_WORLD;
        let is_axis = x == 0;
        let is_major = x.rem_euclid(GRID_MAJOR_EVERY) == 0;
        let color = if is_axis {
            color_u8!(176, 150, 112, 255)
        } else if is_major {
            color_u8!(210, 204, 193, 255)
        } else {
            color_u8!(233, 229, 221, 255)
        };
        let a = world_to_screen(vec2(world_x, bottom), view);
        let b = world_to_screen(vec2(world_x, top), view);
        draw_line(
            a.x,
            a.y,
            b.x,
            b.y,
            if is_axis {
                1.8
            } else if is_major {
                1.15
            } else {
                1.0
            },
            color,
        );
        if is_major || is_axis {
            draw_text(
                &format!("{world_x:.1} m"),
                a.x + 4.0,
                screen_height() - 8.0,
                16.0,
                color_u8!(120, 112, 99, 255),
            );
        }
    }

    let start_y = (bottom / GRID_MINOR_STEP_WORLD).floor() as i32 - 1;
    let end_y = (top / GRID_MINOR_STEP_WORLD).ceil() as i32 + 1;
    for y in start_y..=end_y {
        let world_y = y as f32 * GRID_MINOR_STEP_WORLD;
        let is_axis = y == 0;
        let is_major = y.rem_euclid(GRID_MAJOR_EVERY) == 0;
        let color = if is_axis {
            color_u8!(176, 150, 112, 255)
        } else if is_major {
            color_u8!(210, 204, 193, 255)
        } else {
            color_u8!(233, 229, 221, 255)
        };
        let a = world_to_screen(vec2(left, world_y), view);
        let b = world_to_screen(vec2(right, world_y), view);
        draw_line(
            a.x,
            a.y,
            b.x,
            b.y,
            if is_axis {
                1.8
            } else if is_major {
                1.15
            } else {
                1.0
            },
            color,
        );
        if is_major || is_axis {
            draw_text(
                &format!("{world_y:.1} m"),
                8.0,
                a.y - 4.0,
                16.0,
                color_u8!(120, 112, 99, 255),
            );
        }
    }

    let ground_a = world_to_screen(vec2(left - 4.0, GROUND_Y), view);
    let ground_b = world_to_screen(vec2(right + 4.0, GROUND_Y), view);
    draw_line(
        ground_a.x,
        ground_a.y,
        ground_b.x,
        ground_b.y,
        5.0,
        color_u8!(106, 84, 60, 255),
    );
}

fn draw_finish_line(finish_x: f32, view: &ViewState) {
    let half_h = screen_height() * 0.5 / (PIXELS_PER_METER * view.zoom);
    let top = view.focus.y + half_h;
    let bottom = view.focus.y - half_h;
    let a = world_to_screen(vec2(finish_x, bottom), view);
    let b = world_to_screen(vec2(finish_x, top), view);
    let line_color = color_u8!(236, 127, 43, 255);
    draw_line(a.x, a.y, b.x, b.y, 4.0, line_color);
    draw_text(
        "100 m",
        a.x + 8.0,
        (b.y + 28.0).max(24.0),
        24.0,
        line_color,
    );
}

fn draw_ball_scene(state: &SimulationState, view: &ViewState) {
    if let Some(ball) = &state.ball {
        draw_ball_body(ball, 0.5, view);
    }
}

fn draw_ball_body(ball: &sim_core::BodyState, radius_m: f32, view: &ViewState) {
    let center = world_to_screen(vec2(ball.x, ball.y), view);
    draw_circle(
        center.x,
        center.y,
        radius_m * PIXELS_PER_METER * view.zoom,
        color_u8!(222, 104, 89, 255),
    );
    draw_circle_lines(
        center.x,
        center.y,
        radius_m * PIXELS_PER_METER * view.zoom,
        2.0,
        color_u8!(130, 55, 45, 255),
    );
}

fn draw_robot_scene(sim: &SharedSimulation, _state: &SimulationState, view: &ViewState) {
    let (segments, markers, center_of_mass, state) = {
        let Ok(sim) = sim.lock() else {
            return;
        };
        (
            sim.robot_segments(),
            sim.joint_markers(),
            sim.robot_center_of_mass(),
            sim.state(),
        )
    };

    if let Some(ball) = &state.ball {
        draw_ball_body(ball, 0.2, view);
    }

    for (idx, segment) in segments.iter().enumerate() {
        let a = world_to_screen(vec2(segment.0[0], segment.0[1]), view);
        let b = world_to_screen(vec2(segment.1[0], segment.1[1]), view);
        let color = if idx == 0 {
            color_u8!(188, 88, 70, 255)
        } else if idx < 3 {
            color_u8!(84, 123, 160, 255)
        } else {
            color_u8!(123, 84, 160, 255)
        };
        draw_line(a.x, a.y, b.x, b.y, 20.0 * view.zoom, color);
        draw_circle(a.x, a.y, 8.0 * view.zoom, color_u8!(43, 52, 69, 255));
        draw_circle(b.x, b.y, 8.0 * view.zoom, color_u8!(43, 52, 69, 255));
    }

    for (joint_name, point) in markers {
        if let Some(joint) = state.joints.get(&joint_name) {
            let screen = world_to_screen(vec2(point[0], point[1]), view);
            let text = format!("{:.1}°", joint.angle.to_degrees());
            let dims = measure_text(&text, None, 18, 1.0);
            let text_x = if joint_name.starts_with("left_") {
                screen.x - dims.width - 10.0
            } else {
                screen.x + 10.0
            };
            draw_text(
                &text,
                text_x,
                screen.y - 10.0,
                18.0,
                color_u8!(34, 40, 49, 255),
            );
        }
    }

    if let Some(com) = center_of_mass {
        let screen = world_to_screen(vec2(com[0], com[1]), view);
        let projection = world_to_screen(vec2(com[0], GROUND_Y - 0.12), view);
        let com_color = color_u8!(34, 148, 83, 255);
        let com_outline = color_u8!(20, 93, 52, 255);
        draw_circle(screen.x, screen.y, 6.0 * view.zoom, com_color);
        draw_circle_lines(screen.x, screen.y, 10.0 * view.zoom, 2.0, com_outline);
        draw_circle(projection.x, projection.y, 5.0 * view.zoom, com_color);
        draw_circle_lines(
            projection.x,
            projection.y,
            8.0 * view.zoom,
            2.0,
            com_outline,
        );
    }
}

fn draw_overlay(
    state: &SimulationState,
    view: &ViewState,
    controls: &ControlPanelState,
    fonts: &AppFonts,
) {
    draw_rectangle(24.0, 24.0, 430.0, 300.0, Color::new(1.0, 1.0, 1.0, 0.9));
    let title = match state.scene {
        SceneKind::Robot => "Robot mode",
        SceneKind::Ball => "Ball smoke-test",
    };
    draw_text(title, 40.0, 58.0, 32.0, color_u8!(34, 40, 49, 255));
    draw_text(&format!("time: {:.2}s", state.time), 40.0, 92.0, 24.0, color_u8!(57, 62, 70, 255));
    draw_text(&format!("mode: {}", state.mode), 40.0, 120.0, 24.0, color_u8!(57, 62, 70, 255));
    draw_text(
        &format!(
            "walk: {} {}  {:.2}/{:.2} m/s",
            if state.walk_enabled { "on" } else { "off" },
            if state.walk_direction >= 0.0 { "->" } else { "<-" },
            state.walk_speed,
            state.walk_target_speed
        ),
        40.0,
        148.0,
        24.0,
        color_u8!(57, 62, 70, 255),
    );

    let mut y = 182.0;
    for name in ["right_hip", "right_knee", "left_hip", "left_knee"] {
        if let Some(joint) = state.joints.get(name) {
            draw_text(
                &format!(
                    "{name}: {:.1}° -> {:.1}°",
                    joint.angle.to_degrees(),
                    joint.target.to_degrees()
                ),
                40.0,
                y,
                22.0,
                color_u8!(50, 61, 79, 255),
            );
            y += 24.0;
        }
    }

    draw_text(
        &format!("contacts: L={} R={}", state.contacts.left_foot, state.contacts.right_foot),
        40.0,
        y + 8.0,
        22.0,
        color_u8!(50, 61, 79, 255),
    );
    y += 40.0;
    draw_text(
        &format!("zoom: {:.2}x", view.zoom),
        40.0,
        y + 8.0,
        20.0,
        color_u8!(70, 78, 90, 255),
    );
    draw_text(
        if view.follow_robot {
            "wheel zoom, drag pan, Left/Right walk, S stop, F/Space center robot, R reset"
        } else {
            "wheel zoom, drag pan, Left/Right walk, S stop, F/Space recenter robot, R reset"
        },
        40.0,
        y + 28.0,
        18.0,
        color_u8!(103, 110, 121, 255),
    );

    let status_y = y + 58.0;
    if controls.robofest_finish_x.is_some() {
        draw_text(
            &format!(
                "ROBOFEST 2026: {:.2}s | robot={} | ball={}",
                controls.robofest_elapsed_time,
                controls.robofest_robot_reached,
                controls.robofest_ball_reached
            ),
            40.0,
            status_y,
            20.0,
            color_u8!(57, 62, 70, 255),
        );
    }

    if let Some(message) = &controls.robofest_result_message {
        let box_w = 560.0;
        let box_h = 140.0;
        let x = (screen_width() - box_w) * 0.5;
        let y = 48.0;
        draw_rectangle(x, y, box_w, box_h, Color::new(1.0, 0.98, 0.91, 0.95));
        draw_rectangle_lines(x, y, box_w, box_h, 3.0, color_u8!(220, 146, 42, 255));
        draw_text_with_optional_font(
            "ROBOFEST 2026",
            x + 20.0,
            y + 36.0,
            34,
            color_u8!(125, 72, 16, 255),
            fonts.cyrillic.as_ref(),
        );
        draw_text_with_optional_font(
            message,
            x + 20.0,
            y + 78.0,
            24,
            color_u8!(78, 54, 24, 255),
            fonts.cyrillic.as_ref(),
        );
        draw_text_with_optional_font(
            "Нажмите Start для новой попытки",
            x + 20.0,
            y + 112.0,
            22,
            color_u8!(98, 79, 52, 255),
            fonts.cyrillic.as_ref(),
        );
    }
}

fn draw_text_with_optional_font(
    text: &str,
    x: f32,
    y: f32,
    font_size: u16,
    color: Color,
    font: Option<&Font>,
) {
    if let Some(font) = font {
        draw_text_ex(
            text,
            x,
            y,
            TextParams {
                font: Some(font),
                font_size,
                color,
                ..Default::default()
            },
        );
    } else {
        draw_text(text, x, y, font_size as f32, color);
    }
}

fn world_to_screen(world: Vec2, view: &ViewState) -> Vec2 {
    let center_x = screen_width() * 0.5;
    let center_y = screen_height() * 0.5;
    let relative = world - view.focus;
    vec2(
        center_x + relative.x * PIXELS_PER_METER * view.zoom,
        center_y - relative.y * PIXELS_PER_METER * view.zoom,
    )
}

impl Default for ControlPanelState {
    fn default() -> Self {
        Self {
            right_hip: 0.0,
            right_knee: 0.0,
            left_hip: 0.0,
            left_knee: 0.0,
            right_hip_zero: 0.0,
            right_knee_zero: 0.0,
            left_hip_zero: 0.0,
            left_knee_zero: 0.0,
            servo_kp: 20.0,
            servo_ki: 0.0,
            servo_kd: 0.0,
            servo_max_torque: 2.55,
            suspend_height: 0.8,
            motion_frame_ms: 300.0,
            motion_frames: Vec::new(),
            motion_json: "[]".to_owned(),
            motion_loop: false,
            motion_repeat_delay_ms: 0.0,
            robofest_active: false,
            robofest_start_time: 0.0,
            robofest_elapsed_time: 0.0,
            robofest_finish_x: None,
            robofest_robot_reached: false,
            robofest_ball_reached: false,
            robofest_result_message: None,
            initialized: false,
        }
    }
}

impl ControlPanelState {
    fn sync_from_state(&mut self, state: &SimulationState) {
        if let Some(zero) = state.servo_zeros.get("right_hip") {
            self.right_hip_zero = *zero;
        }
        if let Some(zero) = state.servo_zeros.get("right_knee") {
            self.right_knee_zero = *zero;
        }
        if let Some(zero) = state.servo_zeros.get("left_hip") {
            self.left_hip_zero = *zero;
        }
        if let Some(zero) = state.servo_zeros.get("left_knee") {
            self.left_knee_zero = *zero;
        }
        if let Some(joint) = state.joints.get("right_hip") {
            self.right_hip = joint.target - self.right_hip_zero;
        }
        if let Some(joint) = state.joints.get("right_knee") {
            self.right_knee = joint.target - self.right_knee_zero;
        }
        if let Some(joint) = state.joints.get("left_hip") {
            self.left_hip = joint.target - self.left_hip_zero;
        }
        if let Some(joint) = state.joints.get("left_knee") {
            self.left_knee = joint.target - self.left_knee_zero;
        }
        self.initialized = true;
    }
}

impl Default for ExternalControlState {
    fn default() -> Self {
        Self {
            last_external_command: None,
        }
    }
}

fn mark_external_control(external_control: &SharedExternalControl) {
    if let Ok(mut state) = external_control.lock() {
        state.last_external_command = Some(Instant::now());
    }
}

fn external_control_active(external_control: &SharedExternalControl) -> bool {
    external_control
        .lock()
        .ok()
        .and_then(|state| state.last_external_command)
        .map(|last| last.elapsed() < Duration::from_secs(2))
        .unwrap_or(false)
}


