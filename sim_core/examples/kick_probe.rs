use sim_core::{Simulation, SimulationConfig};
fn main() {
    let mut c = SimulationConfig::load_from_file("robot_config.toml").unwrap();
    if let Some(x) = std::env::args().nth(1) {
        c.robot.ball_spawn_offset_x_m = x.parse().unwrap();
    }
    let mut sim = Simulation::from_config(c);
    let settle: usize = std::env::args()
        .nth(2)
        .unwrap_or("0".into())
        .parse()
        .unwrap();
    for _ in 0..settle {
        sim.step_fixed();
    }
    sim.start_automatic_walk().unwrap();
    let mut history = std::collections::VecDeque::new();
    let mut min_height = 10.0_f32;
    let mut max_tilt = 0.0_f32;
    let mut max_ball_speed = 0.0_f32;
    let mut fell = false;
    for _ in 0..72000 {
        sim.step_fixed();
        let s = sim.state();
        let b = s.base.as_ref().unwrap();
        let ball = s.ball.as_ref().unwrap();
        if std::env::var_os("KICK_TRACE").is_some() {
            history.push_back(serde_json::to_string(&s).unwrap());
            if history.len() > 180 {
                history.pop_front();
            }
        }
        min_height = min_height.min(b.y);
        max_tilt = max_tilt.max(b.angle.abs());
        max_ball_speed = max_ball_speed.max(ball.vx);
        if b.y < 1.15 || b.angle.abs() > 0.6 || !s.walk_enabled {
            fell = true;
            break;
        }
    }
    if fell && std::env::var_os("KICK_TRACE").is_some() {
        std::fs::write(
            "target/kick-fall-trace.jsonl",
            history.into_iter().collect::<Vec<_>>().join("\n"),
        )
        .unwrap();
    }
    let s = sim.state();
    println!(
        "{}",
        serde_json::json!({"time":s.time,"base":s.base,"ball":s.ball,"fell":fell,"min_height":min_height,"max_tilt":max_tilt,"max_ball_speed":max_ball_speed})
    );
}
