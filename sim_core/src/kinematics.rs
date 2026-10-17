#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegSide {
    Left,
    Right,
}

impl LegSide {
    pub fn sign(self) -> f32 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LegIkSolution {
    pub hip: f32,
    pub knee: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FiveLinkKinematics {
    pub torso_length: f32,
    pub thigh_length: f32,
    pub shin_length: f32,
}

impl FiveLinkKinematics {
    // Match Rapier's enhanced-determinism math; platform CRT trig can change
    // touchdown timing and send the point-foot gait into a different orbit.
    pub fn pelvis_from_torso(&self, torso_center: [f32; 2], torso_angle: f32) -> [f32; 2] {
        [
            torso_center[0] + libm::sinf(torso_angle) * self.torso_length * 0.5,
            torso_center[1] - libm::cosf(torso_angle) * self.torso_length * 0.5,
        ]
    }

    pub fn foot_from_angles(
        &self,
        pelvis: [f32; 2],
        torso_angle: f32,
        hip_angle: f32,
        knee_angle: f32,
    ) -> [f32; 2] {
        let thigh_abs = torso_angle + hip_angle;
        let shin_abs = thigh_abs + knee_angle;
        [
            pelvis[0] + libm::sinf(thigh_abs) * self.thigh_length + libm::sinf(shin_abs) * self.shin_length,
            pelvis[1] - libm::cosf(thigh_abs) * self.thigh_length - libm::cosf(shin_abs) * self.shin_length,
        ]
    }

    pub fn solve_leg_ik(
        &self,
        side: LegSide,
        pelvis: [f32; 2],
        torso_angle: f32,
        foot_target: [f32; 2],
    ) -> Option<LegIkSolution> {
        let dx = foot_target[0] - pelvis[0];
        let dy_down = pelvis[1] - foot_target[1];
        let distance_sq = dx * dx + dy_down * dy_down;
        let reach_min = (self.thigh_length - self.shin_length).abs() + 1e-4;
        let reach_max = (self.thigh_length + self.shin_length) - 1e-4;
        let distance = libm::sqrtf(distance_sq).clamp(reach_min, reach_max);
        let scale = if distance_sq > 1e-6 { distance / libm::sqrtf(distance_sq) } else { 0.0 };
        let dx = dx * scale;
        let dy_down = dy_down * scale;
        let cos_knee = ((dx * dx + dy_down * dy_down) - self.thigh_length.powi(2) - self.shin_length.powi(2))
            / (2.0 * self.thigh_length * self.shin_length);
        let cos_knee = cos_knee.clamp(-1.0, 1.0);
        let knee_mag = libm::acosf(cos_knee);
        let knee = side.sign() * knee_mag;
        let hip_world = libm::atan2f(dx, dy_down)
            - libm::atan2f(
                self.shin_length * libm::sinf(knee),
                self.thigh_length + self.shin_length * libm::cosf(knee),
            );
        Some(LegIkSolution {
            hip: normalize_angle(hip_world - torso_angle),
            knee: normalize_angle(knee),
        })
    }
}

fn normalize_angle(angle: f32) -> f32 {
    let mut wrapped = angle;
    while wrapped > std::f32::consts::PI {
        wrapped -= std::f32::consts::TAU;
    }
    while wrapped < -std::f32::consts::PI {
        wrapped += std::f32::consts::TAU;
    }
    wrapped
}
