use sim_core::{Simulation, SimulationConfig};

#[test]
fn forward_walk_remains_upright_from_different_start_times() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../robot_config.toml");
    for settle_steps in [0, 120, 600, 1200] {
        let mut config = SimulationConfig::load_from_file(path).unwrap();
        // Keep the original clear-track scenario; fast ball strikes have a separate regression.
        config.robot.ball_spawn_offset_x_m = 1000.0;
        let mut sim = Simulation::from_config(config);
        for _ in 0..settle_steps {
            sim.step_fixed();
        }
        let start_x = sim.state().base.unwrap().x;
        sim.set_target_direction(1.0);
        sim.set_walk_target_speed(0.25);
        sim.set_directional_walk_enabled(true);
        let mut max_tilt = 0.0_f32;
        let mut min_height = f32::MAX;
        let mut left_lift = 0.0_f32;
        let mut right_lift = 0.0_f32;
        let mut last_x = start_x;
        let mut previous_phase = String::new();
        let mut last_landing = None;
        let mut step_lengths = Vec::new();
        for frame in 0..14400 {
            sim.step_fixed();
            let state = sim.state();
            let base = state.base.unwrap();
            if state.walk_support_state != previous_phase
                && state.walk_support_state.starts_with("DoubleSupport") {
                let foot = if state.walk_support_state.ends_with("Left") {
                    state.left_foot.unwrap()
                } else { state.right_foot.unwrap() };
                if let Some(last) = last_landing {
                    if frame > 1200 { step_lengths.push(foot[0] - last); }
                }
                last_landing = Some(foot[0]);
            }
            previous_phase = state.walk_support_state.clone();
            assert!(state.walk_enabled, "walk stopped after settling {settle_steps} steps");
            assert!((base.x - last_x).abs() < 0.05, "position discontinuity");
            last_x = base.x;
            max_tilt = max_tilt.max(base.angle.abs());
            min_height = min_height.min(base.y);
            left_lift = left_lift.max(state.left_foot.unwrap()[1]);
            right_lift = right_lift.max(state.right_foot.unwrap()[1]);
            for name in ["left_knee", "right_knee"] {
                assert!(state.joints[name].target <= 0.001, "knee bends backwards");
            }
        }
        let distance = last_x - start_x;
        println!("settle={settle_steps}: distance={distance:.2}m, tilt={:.1}deg, height={min_height:.3}m", max_tilt.to_degrees());
        let mean_step = step_lengths.iter().sum::<f32>() / step_lengths.len() as f32;
        println!("mean measured step={mean_step:.3}m, landings={}", step_lengths.len());
        assert!(step_lengths.len() > 100);
        assert!((0.18..0.24).contains(&mean_step), "step must stay near 20cm");
        assert!(distance > 30.0, "standing still is not walking");
        assert!(max_tilt < 25.0_f32.to_radians());
        assert!(min_height > 1.2);
        assert!(left_lift > 0.015 && right_lift > 0.015, "both feet must swing");
    }
}
