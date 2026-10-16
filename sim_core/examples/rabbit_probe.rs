//! Headless contact-model experiment. Does not reset after a fall.
use sim_core::{Simulation, SimulationConfig};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = SimulationConfig::load_from_file("robot_config.toml").unwrap();
    let mut sim = Simulation::from_config(config);
    let json = std::fs::read_to_string(
        args.iter()
            .find(|a| a.ends_with(".json"))
            .map(String::as_str)
            .unwrap_or("motions/rabbit-20cm-030.json"),
    )
    .unwrap();
    let settle = args
        .iter()
        .position(|s| s == "--settle")
        .map(|i| args[i + 1].parse::<usize>().unwrap())
        .unwrap_or(0);
    for _ in 0..settle {
        sim.step_fixed();
    }
    if args.iter().any(|a| a == "--orbit") {
        sim.reset_to_rabbit_orbit(&json).unwrap();
    } else {
        sim.load_rabbit_trajectory(&json).unwrap();
    }
    let mut last_steps = 0;
    let mut last_foot = None;
    let mut lengths = Vec::new();
    let mut max_tilt = 0.0_f32;
    let mut min_y = 10.0_f32;
    let mut steady_x = 0.0_f32;
    let mut steady_steps = 0;
    let mut steady_lift = 0.0_f32;
    for frame in 0..14400 {
        if frame == 3600 && args.iter().any(|a| a == "--stop") {
            sim.disable_rabbit_controller();
        }
        if frame == 3600 && args.iter().any(|a| a == "--accelerate") {
            sim.set_rabbit_nominal_speed(0.5).unwrap();
        }
        sim.step_fixed();
        let s = sim.state();
        let b = s.base.as_ref().unwrap();
        if frame == 7200 {
            steady_x = b.x;
            steady_steps = s.planned_steps;
        }
        if frame >= 7200 {
            steady_lift = steady_lift
                .max(s.left_foot.unwrap()[1])
                .max(s.right_foot.unwrap()[1]);
        }
        max_tilt = max_tilt.max(b.angle.abs());
        min_y = min_y.min(b.y);
        if s.planned_steps > last_steps {
            let foot = if s.walk_support_state.ends_with("Left") {
                s.left_foot.unwrap()
            } else {
                s.right_foot.unwrap()
            };
            if let Some(last) = last_foot {
                lengths.push(foot[0] - last);
            }
            last_foot = Some(foot[0]);
            last_steps = s.planned_steps;
        }
        if !s.walk_enabled && !args.iter().any(|a| a == "--stop") {
            break;
        }
    }
    let s = sim.state();
    let b = s.base.unwrap();
    println!(
        "{}",
        serde_json::json!({"time_s":s.time,"x_m":b.x,"height_m":b.y,"steady_speed_mps":(b.x-steady_x)/60.0,"steady_cadence_spm":last_steps-steady_steps,"steady_peak_foot_height_m":steady_lift,"min_height_m":min_y,"max_pitch_deg":max_tilt.to_degrees(),"steps":last_steps,"mean_step_m":if lengths.is_empty(){0.0}else{lengths.iter().sum::<f32>()/lengths.len() as f32},"walking":s.walk_enabled})
    );
}
