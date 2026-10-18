//! Same configuration and simulation-tick commands on every platform.
use sim_core::{Simulation, SimulationConfig};
fn main() {
    let mut sim = Simulation::from_config(SimulationConfig::default());
    let mut states = Vec::new();
    for tick in 0..21600 {
        if tick == 0 || tick == 14400 { sim.start_automatic_walk().unwrap(); }
        if tick == 7200 { sim.set_directional_walk_enabled(false); }
        sim.step_fixed();
        if (tick+1)%120 == 0 { states.push(sim.state()); }
    }
    println!("{}", serde_json::to_string(&states).unwrap());
}
