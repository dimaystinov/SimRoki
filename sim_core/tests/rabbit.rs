use nalgebra::SVector;
use sim_core::{
    rabbit::{RabbitModel, RabbitTrajectory},
    SimulationConfig,
};
#[test]
fn rust_dynamics_match_independent_symbolic_lagrangian() {
    let c = SimulationConfig::load_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../robot_config.toml"
    ))
    .unwrap();
    let model = RabbitModel::new(&c);
    let f: serde_json::Value =
        serde_json::from_str(include_str!("../../planner/dynamics-fixture.json")).unwrap();
    let q = SVector::<f64, 5>::from_fn(|i, _| f["q"][i].as_f64().unwrap());
    let v = SVector::<f64, 5>::from_fn(|i, _| f["v"][i].as_f64().unwrap());
    let m = model.mass(&q);
    let h = model.bias(&q, &v);
    for i in 0..5 {
        assert!((h[i] - f["bias"][i].as_f64().unwrap()).abs() < 1e-7);
        for j in 0..5 {
            assert!((m[(i, j)] - f["mass"][i][j].as_f64().unwrap()).abs() < 1e-7);
        }
    }
}
#[test]
fn orbit_is_monotone_and_controller_recovers_nominal_torques() {
    let c = SimulationConfig::load_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../robot_config.toml"
    ))
    .unwrap();
    let model = RabbitModel::new(&c);
    let p =
        RabbitTrajectory::from_json(include_str!("../../motions/rabbit-20cm-030.json")).unwrap();
    model.validate_trajectory(&p).unwrap();
    for i in 1..1000 {
        let phase = i as f64 / 1000.0;
        let (q, v, u) = p.sample(phase);
        assert!((model.phase(&q, &p) - phase).abs() < 1e-7);
        let actual = model.control(&q, &v, &p, 100., 20.).unwrap();
        assert!(
            (actual - u).amax() < 0.01,
            "control differs at {phase}: {actual:?} vs {u:?}"
        );
    }
}
#[test]
fn rejects_malformed_or_infeasible_plan() {
    assert!(RabbitTrajectory::from_json("{}").is_err());
    let mut p: serde_json::Value =
        serde_json::from_str(include_str!("../../motions/rabbit-20cm-030.json")).unwrap();
    p["duration_s"] = serde_json::json!(-1.);
    assert!(RabbitTrajectory::from_json(&p.to_string()).is_err());
}

#[test]
fn planned_walk_starts_from_rest_and_stops_without_resetting() {
    use sim_core::Simulation;
    for settle in [0, 120, 600, 1200] {
        let c = SimulationConfig::load_from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../robot_config.toml"
        ))
        .unwrap();
        let mut sim = Simulation::from_config(c);
        for _ in 0..settle {
            sim.step_fixed();
        }
        sim.load_rabbit_trajectory(include_str!("../../motions/rabbit-20cm-030.json"))
            .unwrap();
        let mut max_pitch = 0.0_f32;
        let mut min_height = 10.0_f32;
        let mut steps = 0;
        let mut previous = sim.state().base.unwrap().x;
        for frame in 0..14400 {
            if frame == 7200 {
                sim.disable_rabbit_controller();
            }
            sim.step_fixed();
            let state = sim.state();
            let base = state.base.unwrap();
            assert!((base.x - previous).abs() < 0.05, "position jump");
            previous = base.x;
            max_pitch = max_pitch.max(base.angle.abs());
            min_height = min_height.min(base.y);
            steps = steps.max(state.planned_steps);
            if frame < 7200 {
                assert!(
                    state.walk_enabled,
                    "fall during walk, settle={settle}, frame={frame}"
                );
            }
            if frame > 7560 {
                assert!(!state.walk_enabled, "stop must finish on contact");
            }
            for joint in state.joints.values() {
                assert!(joint.torque.abs() <= 2.55001);
            }
        }
        let state = sim.state();
        let base = state.base.unwrap();
        println!("RABBIT settle={settle}: x={:.2}m steps={steps} max_pitch={:.2}deg min_height={min_height:.3}",base.x,max_pitch.to_degrees());
        assert!(base.x > 5.0 && steps > 30);
        assert!(max_pitch.to_degrees() < 12.0 && min_height > 1.25);
        assert!(base.vx.abs() < 0.02, "must come to rest");
    }
}

#[test]
fn phase_advances_faster_when_body_moves_faster() {
    let c = SimulationConfig::load_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../robot_config.toml"
    ))
    .unwrap();
    let model = RabbitModel::new(&c);
    let p =
        RabbitTrajectory::from_json(include_str!("../../motions/rabbit-20cm-030.json")).unwrap();
    for s in [0.2, 0.4, 0.6, 0.8] {
        let (q, v, _) = p.sample(s);
        let slow = model.phase(&(q + v * 0.001), &p) - s;
        let fast = model.phase(&(q + v * 0.002), &p) - s;
        assert!(slow > 0. && (fast / slow - 2.).abs() < 0.02);
    }
}

#[test]
fn changing_nominal_speed_keeps_contacts_and_motor_limits() {
    use sim_core::Simulation;
    let c = SimulationConfig::load_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../robot_config.toml"
    ))
    .unwrap();
    let mut sim = Simulation::from_config(c);
    sim.load_rabbit_trajectory(include_str!("../../motions/rabbit-20cm-030.json"))
        .unwrap();
    for frame in 0..14400 {
        if frame == 3600 {
            sim.set_rabbit_nominal_speed(0.5).unwrap();
        }
        if frame == 7200 {
            sim.set_rabbit_nominal_speed(0.3).unwrap();
        }
        sim.step_fixed();
        let s = sim.state();
        let b = s.base.unwrap();
        assert!(
            s.walk_enabled && b.y > 1.25 && b.angle.abs() < 0.25,
            "speed transition fell at {frame}"
        );
        assert!(s.joints.values().all(|j| j.torque.abs() <= 2.55001));
    }
    assert!(sim.state().planned_steps > 60);
}

#[test]
fn higher_faster_orbit_increases_actual_speed_and_foot_lift() {
    use sim_core::Simulation;
    for settle in [120, 600, 1200] {
        let c = SimulationConfig::load_from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../robot_config.toml"
        ))
        .unwrap();
        let mut sim = Simulation::from_config(c);
        for _ in 0..settle {
            sim.step_fixed();
        }
        sim.load_rabbit_trajectory(include_str!("../../motions/rabbit-20cm-060-high.json"))
            .unwrap();
        let mut x = 0.0;
        let mut lift = 0.0_f32;
        for frame in 0..14400 {
            sim.step_fixed();
            let s = sim.state();
            let b = s.base.unwrap();
            assert!(s.walk_enabled && b.y > 1.25 && b.angle.abs() < 0.22, "settle={settle} frame={frame} enabled={} height={} pitch={} steps={}", s.walk_enabled, b.y, b.angle, s.planned_steps);
            if frame == 7200 {
                x = b.x;
            }
            if frame >= 7200 {
                lift = lift
                    .max(s.left_foot.unwrap()[1])
                    .max(s.right_foot.unwrap()[1]);
            }
        }
        let speed = (sim.state().base.unwrap().x - x) / 60.0;
        assert!(speed > 0.21, "actual speed {speed}");
        assert!(lift > 0.025, "actual swing height {lift}");
    }
}
