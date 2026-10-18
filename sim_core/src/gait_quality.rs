//! Competition heuristics, not a biomechanical or clinical human-likeness metric.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaitQualityReport {
    pub score: f32,
    pub eligible: bool,
    pub status: String,
    pub locomotion: String,
    pub reason: Option<String>,
    pub landings: u32,
    pub alternation: f32,
    pub flight_seconds: f32,
    pub forbidden_support_seconds: f32,
}

#[derive(Default)]
pub(crate) struct GaitQuality {
    time: f32,
    posture_sum: f32,
    forbidden: f32,
    forbidden_streak: f32,
    low_streak: f32,
    flight: f32,
    flight_streak: f32,
    has_flight: bool,
    off: [f32; 2],
    last_side: Option<usize>,
    landings: u32,
    alternating: u32,
    moving_without_step: f32,
    reason: Option<String>,
}

impl GaitQuality {
    pub fn sample(
        &mut self,
        dt: f32,
        tilt: f32,
        hip_height_ratio: f32,
        speed: f32,
        feet: [bool; 2],
        forbidden: bool,
        suspended: bool,
    ) {
        self.time += dt;
        // Forward lean is allowed, including running; only deep crouching or
        // near-horizontal posture is penalized. Height is normalized to leg length.
        let upright = ((70.0 - tilt.abs().to_degrees()) / 35.0).clamp(0.0, 1.0);
        let tall = ((hip_height_ratio - 0.35) / 0.30).clamp(0.0, 1.0);
        self.posture_sum += dt * upright.min(tall);
        if forbidden {
            self.forbidden += dt;
            self.forbidden_streak += dt;
        } else {
            self.forbidden_streak = 0.0;
        }
        if hip_height_ratio < 0.45 || tilt.abs().to_degrees() > 70.0 {
            self.low_streak += dt;
        } else {
            self.low_streak = 0.0;
        }
        if feet == [false, false] {
            self.flight += dt;
            self.flight_streak += dt;
            if self.flight_streak >= 0.03 && self.landings >= 2 {
                self.has_flight = true;
            }
        } else {
            self.flight_streak = 0.0;
        }

        let mut landed = Vec::new();
        for (side, grounded) in feet.into_iter().enumerate() {
            if grounded {
                // Ignore solver chatter and the initial placement on the floor.
                if self.off[side] >= 0.04 {
                    landed.push(side);
                }
                self.off[side] = 0.0;
            } else {
                self.off[side] += dt;
            }
        }
        if !landed.is_empty() {
            self.moving_without_step = 0.0;
            let side = if landed.len() == 2 { 2 } else { landed[0] };
            if self
                .last_side
                .is_some_and(|last| last < 2 && side < 2 && last != side)
            {
                self.alternating += 1;
            }
            self.landings += 1;
            self.last_side = Some(side);
        } else if speed.abs() > 0.2 {
            self.moving_without_step += dt;
        } else {
            self.moving_without_step = 0.0;
        }

        if self.reason.is_none() {
            self.reason = if suspended {
                Some("External suspension")
            } else if self.forbidden_streak >= 0.15 || self.forbidden >= 0.5 {
                Some("Body/knee support on ground")
            } else if self.low_streak >= 0.5 {
                Some("Crawling or horizontal torso")
            } else if self.moving_without_step >= 3.0 {
                Some("Travel without stepping")
            } else if self.landings >= 8 && self.alternation() < 0.6 {
                Some("Legs do not alternate")
            } else {
                None
            }
            .map(str::to_owned);
        }
    }

    fn alternation(&self) -> f32 {
        if self.landings < 2 {
            1.0
        } else {
            self.alternating as f32 / (self.landings - 1) as f32
        }
    }

    pub fn report(&self) -> GaitQualityReport {
        let eligible = self.reason.is_none() && self.landings >= 8;
        let score = if self.time == 0.0 {
            0.0
        } else {
            40.0 * self.posture_sum / self.time
                + 30.0 * self.alternation()
                + 30.0 * (1.0 - self.forbidden / self.time)
        }
        .clamp(0.0, 100.0);
        GaitQualityReport {
            score: if self.reason.is_some() {
                score.min(49.0)
            } else {
                score
            },
            eligible,
            status: if self.reason.is_some() {
                "invalid"
            } else if eligible {
                "eligible"
            } else {
                "observing"
            }
            .into(),
            locomotion: if self.landings < 8 {
                "observing"
            } else if self.has_flight {
                "running"
            } else {
                "walking"
            }
            .into(),
            reason: self.reason.clone(),
            landings: self.landings,
            alternation: self.alternation(),
            flight_seconds: self.flight,
            forbidden_support_seconds: self.forbidden,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn trial(running: bool, hopping: bool, forbidden: bool, low: bool) -> GaitQualityReport {
        let mut q = GaitQuality::default();
        for tick in 0..1200 {
            let phase = tick % 60;
            let feet = if running && phase % 30 >= 20 {
                [false, false]
            } else if hopping {
                [true, false]
            } else if phase < 30 {
                [true, false]
            } else {
                [false, true]
            };
            q.sample(
                1.0 / 120.0,
                if low { 1.4 } else { 0.4 },
                if low { 0.3 } else { 0.9 },
                2.0,
                feet,
                forbidden,
                false,
            );
        }
        q.report()
    }
    #[test]
    fn walking_and_running_are_eligible() {
        for run in [false, true] {
            let r = trial(run, false, false, false);
            assert!(r.eligible, "{r:?}");
            assert!(r.score > 95.0);
            assert_eq!(r.locomotion, if run { "running" } else { "walking" });
        }
    }
    #[test]
    fn body_support_crawling_and_one_leg_hops_are_invalid() {
        for r in [
            trial(false, false, true, false),
            trial(false, false, false, true),
            trial(true, true, false, false),
        ] {
            assert!(!r.eligible);
            assert_eq!(r.status, "invalid");
        }
    }
    #[test]
    fn mean_score_uses_all_samples_and_their_duration() {
        let mut q = GaitQuality::default();
        // 1 s with full posture credit, then 3 s with half posture credit:
        // 40 * (1 + 3*0.5)/4 + 30 + 30 = 85.
        for _ in 0..120 {
            q.sample(1.0 / 120.0, 0.0, 0.9, 0.0, [true, true], false, false);
        }
        for _ in 0..180 {
            q.sample(
                1.0 / 60.0,
                52.5_f32.to_radians(),
                0.9,
                0.0,
                [true, true],
                false,
                false,
            );
        }
        let r = q.report();
        assert!((r.score - 85.0).abs() < 0.01, "{r:?}");
        assert!(!r.eligible, "standing cannot qualify as locomotion");
    }

    #[test]
    fn sliding_is_invalid() {
        let r = trial(false, true, false, false);
        assert_eq!(r.reason.as_deref(), Some("Travel without stepping"));
    }
}
