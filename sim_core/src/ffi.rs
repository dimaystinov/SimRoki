use crate::{
    GaitCommand, JointName, MotionSequenceCommand, PoseCommand, RlResetCommand, RlStepCommand, SceneKind,
    ServoTargets, Simulation, WalkConfigCommand, WalkDirectionCommand,
};
use serde::Serialize;
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_float, c_int},
    ptr,
    sync::{Mutex, OnceLock},
};

fn error_store() -> &'static Mutex<String> {
    static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(message: impl Into<String>) {
    if let Ok(mut slot) = error_store().lock() {
        *slot = message.into();
    }
}

fn clear_last_error() {
    if let Ok(mut slot) = error_store().lock() {
        slot.clear();
    }
}

fn with_sim_mut<T>(sim: *mut Simulation, f: impl FnOnce(&mut Simulation) -> Result<T, String>) -> Result<T, String> {
    if sim.is_null() {
        return Err("simulation pointer is null".to_owned());
    }
    let sim_ref = unsafe { &mut *sim };
    f(sim_ref)
}

fn read_c_string(ptr_value: *const c_char, what: &str) -> Result<String, String> {
    if ptr_value.is_null() {
        return Err(format!("{what} pointer is null"));
    }
    let value = unsafe { CStr::from_ptr(ptr_value) };
    value
        .to_str()
        .map(|text| text.to_owned())
        .map_err(|_| format!("{what} is not valid UTF-8"))
}

fn json_from_ptr<T: serde::de::DeserializeOwned>(ptr_value: *const c_char, what: &str) -> Result<T, String> {
    let text = read_c_string(ptr_value, what)?;
    serde_json::from_str(&text).map_err(|err| format!("invalid {what} JSON: {err}"))
}

fn into_c_string(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(text) => text.into_raw(),
        Err(_) => {
            set_last_error("string contains interior NUL byte");
            ptr::null_mut()
        }
    }
}

fn ok_json_ptr<T: Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(json) => {
            clear_last_error();
            into_c_string(json)
        }
        Err(err) => {
            set_last_error(format!("failed to serialize JSON: {err}"));
            ptr::null_mut()
        }
    }
}

fn scene_from_name(name: &str) -> Result<SceneKind, String> {
    match name {
        "robot" => Ok(SceneKind::Robot),
        "ball" => Ok(SceneKind::Ball),
        _ => Err(format!("unknown scene '{name}', expected 'robot' or 'ball'")),
    }
}

fn ffi_success(result: Result<(), String>) -> c_int {
    match result {
        Ok(()) => {
            clear_last_error();
            1
        }
        Err(err) => {
            set_last_error(err);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_create_default() -> *mut Simulation {
    clear_last_error();
    Box::into_raw(Box::new(Simulation::default()))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_create_from_config_path(path: *const c_char) -> *mut Simulation {
    let path = match read_c_string(path, "config path") {
        Ok(path) => path,
        Err(err) => {
            set_last_error(err);
            return ptr::null_mut();
        }
    };
    match Simulation::from_config_file(path) {
        Ok(sim) => {
            clear_last_error();
            Box::into_raw(Box::new(sim))
        }
        Err(err) => {
            set_last_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_destroy(sim: *mut Simulation) {
    if sim.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(sim));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_last_error_message() -> *mut c_char {
    let message = error_store()
        .lock()
        .map(|slot| slot.clone())
        .unwrap_or_else(|_| "failed to lock error store".to_owned());
    into_c_string(message)
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_string_free(text: *mut c_char) {
    if text.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(text));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_step_for_seconds(sim: *mut Simulation, frame_dt: c_float) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.step_for_seconds(frame_dt);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_pause(sim: *mut Simulation) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.pause();
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_resume(sim: *mut Simulation) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.resume();
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_reset_robot(sim: *mut Simulation) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.reset_robot();
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_reset_ball(sim: *mut Simulation) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.reset_ball();
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_save_config(sim: *mut Simulation, path: *const c_char) -> c_int {
    let path = match read_c_string(path, "config path") {
        Ok(path) => path,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| sim.save_config_file(path)))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_scene(sim: *mut Simulation, scene_name: *const c_char) -> c_int {
    let scene_name = match read_c_string(scene_name, "scene name") {
        Ok(name) => name,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        let scene = scene_from_name(&scene_name)?;
        sim.set_scene(scene);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_joint_target(sim: *mut Simulation, joint_name: *const c_char, angle: c_float) -> c_int {
    let joint_name = match read_c_string(joint_name, "joint name") {
        Ok(name) => name,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        let joint = joint_name.parse::<JointName>()?;
        sim.set_joint_target(joint, angle);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_servo_gains(
    sim: *mut Simulation,
    kp: c_float,
    ki: c_float,
    kd: c_float,
    max_torque: c_float,
) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_servo_gains(kp, ki, kd, max_torque);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_robot_suspended(sim: *mut Simulation, suspended: c_int) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_robot_suspended(suspended != 0);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_suspend_clearance(sim: *mut Simulation, clearance: c_float) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_suspend_clearance(clearance);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_servo_zero_to_current_pose(sim: *mut Simulation) -> c_int {
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_servo_zero_to_current_pose();
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_apply_targets_json(sim: *mut Simulation, targets_json: *const c_char) -> c_int {
    let targets = match json_from_ptr::<ServoTargets>(targets_json, "servo targets") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.apply_targets(targets);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_apply_pose_json(sim: *mut Simulation, pose_json: *const c_char) -> c_int {
    let pose = match json_from_ptr::<PoseCommand>(pose_json, "pose command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.apply_pose(pose);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_gait_json(sim: *mut Simulation, gait_json: *const c_char) -> c_int {
    let gait = match json_from_ptr::<GaitCommand>(gait_json, "gait command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_gait(gait);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_motion_sequence_deg_json(sim: *mut Simulation, motion_json: *const c_char) -> c_int {
    let motion = match json_from_ptr::<MotionSequenceCommand>(motion_json, "motion sequence command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_motion_sequence_deg(motion);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_walk_direction_json(sim: *mut Simulation, command_json: *const c_char) -> c_int {
    let command = match json_from_ptr::<WalkDirectionCommand>(command_json, "walk direction command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.set_target_direction(command.direction);
        sim.set_directional_walk_enabled(command.enabled.unwrap_or(true));
        if let Some(speed_mps) = command.speed_mps {
            sim.set_walk_target_speed(speed_mps);
        }
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_set_walk_config_json(sim: *mut Simulation, command_json: *const c_char) -> c_int {
    let command = match json_from_ptr::<WalkConfigCommand>(command_json, "walk config command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return 0;
        }
    };
    ffi_success(with_sim_mut(sim, |sim| {
        sim.update_walk_config(command);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_state_json(sim: *mut Simulation) -> *mut c_char {
    match with_sim_mut(sim, |sim| Ok(sim.state())) {
        Ok(state) => ok_json_ptr(&state),
        Err(err) => {
            set_last_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_rl_observation_json(sim: *mut Simulation) -> *mut c_char {
    match with_sim_mut(sim, |sim| Ok(sim.rl_observation())) {
        Ok(observation) => ok_json_ptr(&observation),
        Err(err) => {
            set_last_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_rl_reset_json(sim: *mut Simulation, command_json: *const c_char) -> *mut c_char {
    let command = if command_json.is_null() {
        RlResetCommand::default()
    } else {
        match json_from_ptr::<RlResetCommand>(command_json, "rl reset command") {
            Ok(value) => value,
            Err(err) => {
                set_last_error(err);
                return ptr::null_mut();
            }
        }
    };
    match with_sim_mut(sim, |sim| Ok(sim.rl_reset_with(command))) {
        Ok(result) => ok_json_ptr(&result),
        Err(err) => {
            set_last_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_rl_step_json(sim: *mut Simulation, command_json: *const c_char) -> *mut c_char {
    let command = match json_from_ptr::<RlStepCommand>(command_json, "rl step command") {
        Ok(value) => value,
        Err(err) => {
            set_last_error(err);
            return ptr::null_mut();
        }
    };
    match with_sim_mut(sim, |sim| Ok(sim.rl_step_deg(command))) {
        Ok(result) => ok_json_ptr(&result),
        Err(err) => {
            set_last_error(err);
            ptr::null_mut()
        }
    }
}
