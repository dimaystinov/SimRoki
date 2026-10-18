use sim_core::{Simulation, SimulationConfig};

#[test]
fn identical_physics_at_different_render_rates_and_frame_stalls() {
    let config = SimulationConfig::default();
    let dt = config.physics.dt;
    let mut reference = Simulation::from_config(config.clone());
    reference.start_automatic_walk().unwrap();
    let mut checkpoints = Vec::new();
    // Each checkpoint is exactly one simulated second, including the ball encounter.
    for _ in 0..60 {
        for _ in 0..120 { reference.step_fixed(); }
        checkpoints.push(serde_json::to_value(reference.state()).unwrap());
    }
    for pattern in [&[1][..], &[2], &[4], &[8], &[1, 7, 2, 30, 4, 15, 61]] {
        let mut sim = Simulation::from_config(config.clone());
        sim.start_automatic_walk().unwrap();
        for (second, expected) in checkpoints.iter().enumerate() {
            let mut remaining = 120;
            let mut index = 0;
            while remaining > 0 {
                let ticks = pattern[index % pattern.len()].min(remaining);
                sim.step_for_seconds(dt * ticks as f32);
                remaining -= ticks;
                index += 1;
            }
            assert_eq!(&serde_json::to_value(sim.state()).unwrap(), expected,
                "physics differs at second {} for render batches {pattern:?}", second+1);
        }
    }
}
