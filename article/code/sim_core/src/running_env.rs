//! Shared RABBIT configuration for desktop, headless checks and Gymnasium.
use crate::Simulation;
use serde::{Deserialize, Serialize};

/// Reset-only perturbations, in absolute link coordinates (left shin to right shin).
/// Forward kinematics preserves joints. These are never applied during recovery.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RabbitStartState {
    pub q_offsets_rad: [f32; 5],
    pub angular_velocity_rad_s: [f32; 5],
    pub linear_velocity_mps: [f32; 2],
}

impl RabbitStartState {
    pub fn validate(&self) -> Result<(), String> {
        if self.q_offsets_rad.iter().any(|x| !x.is_finite() || x.abs()>0.6)
            || self.angular_velocity_rad_s.iter().any(|x| !x.is_finite() || x.abs()>5.0)
            || self.linear_velocity_mps.iter().any(|x| !x.is_finite() || x.abs()>3.0) {
            return Err("invalid start perturbation: angles <=0.6 rad, angular speeds <=5 rad/s, linear speeds <=3 m/s".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
pub struct RabbitResidualStep {
    pub action: [f64;5],
    pub steps: u32,
    #[serde(default)]
    pub torso_impulse_ns: Option<[f32;2]>,
}

#[derive(Serialize)]
pub struct RabbitObservation {
    pub state: crate::SimulationState,
    pub center_of_mass: Option<[f32;2]>,
    pub step_metrics: Option<RabbitStepMetrics>,
    pub recovery: crate::rabbit_recovery::RecoveryStatus,
}

#[derive(Default,Serialize)]
pub struct RabbitStepMetrics {
    pub physics_frames: u32,
    pub double_contact_frames: u32,
    pub wrong_contact_frames: u32,
    pub absolute_work_j: f64,
    pub recovery_frames: u32,
}

pub fn observation(sim:&Simulation)->RabbitObservation {
    RabbitObservation{state:sim.state(),center_of_mass:sim.robot_center_of_mass(),step_metrics:None,recovery:sim.rabbit_recovery_status()}
}

impl RabbitResidualStep {
    pub fn apply(self,sim:&mut Simulation)->Result<RabbitObservation,String> {
        if !(1..=240).contains(&self.steps) {return Err("steps must be 1..240".into());}
        sim.set_rabbit_target_residual([self.action[0],self.action[1],self.action[2],self.action[3]],self.action[4])?;
        if let Some(impulse)=self.torso_impulse_ns {sim.apply_robot_impulse(impulse)?;}
        let mut metrics=RabbitStepMetrics::default();
        for _ in 0..self.steps {
            sim.step_fixed();
            let s=sim.state();
            let recovering=sim.rabbit_recovery_status().active;
            let double=s.contacts.left_foot && s.contacts.right_foot && !recovering;
            let swing=if s.planned_stance_left.unwrap_or(true) {s.contacts.right_foot}else{s.contacts.left_foot};
            let wrong=double || (s.planned_status=="running_stance" && swing);
            metrics.physics_frames+=1;
            metrics.double_contact_frames+=u32::from(double);
            metrics.wrong_contact_frames+=u32::from(wrong);
            metrics.recovery_frames+=u32::from(recovering);
            metrics.absolute_work_j+=s.joints.values().map(|j|(j.torque*j.velocity).abs() as f64).sum::<f64>()*sim.dt() as f64;
            if sim.rabbit_episode_failed() {break;}
        }
        let mut result=observation(sim);
        result.step_metrics=Some(metrics);
        Ok(result)
    }
}

#[derive(Clone, Deserialize)]
pub struct RabbitRunCommand {
    #[serde(default)]
    pub recovery_grace_s: f32,
    #[serde(default)]
    pub recovery_pd: Option<[f32;2]>,
    #[serde(default)]
    pub knp_recovery_policy: Option<crate::rabbit_recovery::RecoveryPolicy>,
    #[serde(default)]
    pub knp_recovery_motion: Option<crate::rabbit_recovery::RecoveryMotion>,
    #[serde(default)]
    pub start_state: Option<RabbitStartState>,
    #[serde(default)]
    pub startup_blend_s: f64,
    #[serde(default)]
    pub stance_phase_feedback: f64,
    #[serde(default)]
    pub knp_phase_sensor_table: Option<crate::rabbit::RabbitLeanSensorTable>,
    pub plan: serde_json::Value,
    pub dt: f32,
    pub servo_max_torque: Option<f32>,
    pub start_from_rest: Option<bool>,
    pub initial_lean: Option<f32>,
    pub stance_reference: crate::rabbit::RabbitReferenceTable,
    pub flight_reference: crate::rabbit::RabbitReferenceTable,
    pub first_flight_reference: Option<crate::rabbit::RabbitReferenceTable>,
    pub first_takeoff_clearance_m: Option<f32>,
    pub stance_kp: f64,
    pub stance_kd: f64,
    pub flight_kp: f64,
    pub flight_kd: f64,
    pub flight_offsets: [f64;4],
    pub push_norm: f32,
    pub push_start: f64,
    pub push_end: f64,
    pub placement_gain: f64,
    pub placement_speed: f64,
    pub com_placement: Option<[f64;3]>,
    pub com_ik: Option<[f64;4]>,
    pub pitch_kp: f64,
    pub pitch_kd: f64,
    pub adaptive_lean: Option<[f64;6]>,
    pub knp_lean_table: Option<crate::rabbit::RabbitLeanTable>,
    pub knp_lean_sensor_table: Option<crate::rabbit::RabbitLeanSensorTable>,
}

pub fn configure_rabbit_run(sim: &mut Simulation, command: RabbitRunCommand) -> Result<(), String> {
    let json = serde_json::to_string(&command.plan).map_err(|error| error.to_string())?;
    sim.set_physics_dt(command.dt)?;
    if let Some(limit)=command.servo_max_torque {
        if !limit.is_finite() || !(0.1..=10.0).contains(&limit) {
            return Err("running servo maximum torque must be 0.1..10 N m".into());
        }
        let (kp,ki,kd,_)=sim.servo_gains();
        sim.set_servo_gains(kp,ki,kd,limit);
    }
    // The running model snapshots the configured torque limit during reset.
    if let Some(start)=command.start_state.as_ref() {
        sim.reset_to_rabbit_running_custom_state(&json,command.initial_lean.unwrap_or(0.0),
            if command.start_from_rest.unwrap_or(false) {0.0} else {1.0},start)?;
    } else if command.start_from_rest.unwrap_or(false) {
        sim.reset_to_rabbit_running_stance_from_rest(&json,command.initial_lean.unwrap_or(0.0))?;
    } else {
        sim.reset_to_rabbit_running_orbit_with_lean(&json,command.initial_lean.unwrap_or(0.0))?;
    }
    sim.set_rabbit_virtual_reference(Some(command.stance_reference),1.0)?;
    sim.set_rabbit_running_flight_reference(Some(command.flight_reference))?;
    sim.set_rabbit_running_first_flight_reference(command.first_flight_reference)?;
    sim.set_rabbit_first_takeoff_clearance(command.first_takeoff_clearance_m.unwrap_or(0.0))?;
    sim.set_rabbit_running_stance_gains(Some((command.stance_kp,command.stance_kd)))?;
    sim.set_rabbit_flight_pd(Some((command.flight_kp,command.flight_kd,command.flight_offsets)))?;
    sim.set_rabbit_stance_push(Some((command.push_norm*2.55,command.push_start,command.push_end)))?;
    sim.set_rabbit_flight_placement(Some((command.placement_gain,command.placement_speed)))?;
    if let Some([gain,offset,lookahead])=command.com_placement {
        sim.set_rabbit_flight_com_placement(Some((gain,offset,lookahead)))?;
    }
    if let Some([offset,lookahead,blend,start])=command.com_ik {
        sim.set_rabbit_flight_com_ik(Some((offset,lookahead,blend,start)))?;
    }
    sim.set_rabbit_flight_pitch(Some((command.pitch_kp,command.pitch_kd)))?;
    if let Some([base,gain,reference_speed,min,max,recovery])=command.adaptive_lean {
        sim.set_rabbit_adaptive_lean(Some((base,gain,reference_speed,min,max,recovery)))?;
    }else{
        sim.set_rabbit_adaptive_lean(None)?;
    }
    sim.set_rabbit_knp_lean_table(command.knp_lean_table)?;
    sim.set_rabbit_knp_lean_sensor_table(command.knp_lean_sensor_table)?;
    sim.set_rabbit_startup_blend(command.startup_blend_s)?;
    sim.set_rabbit_stance_phase_feedback(command.stance_phase_feedback)?;
    sim.set_rabbit_knp_phase_table(command.knp_phase_sensor_table)?;
    sim.set_rabbit_recovery(command.recovery_grace_s,command.knp_recovery_policy)?;
    sim.set_rabbit_recovery_pd(command.recovery_pd)?;
    sim.set_rabbit_recovery_motion(command.knp_recovery_motion)?;
    sim.resume();
    Ok(())
}
