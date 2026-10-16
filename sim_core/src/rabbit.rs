//! Point-foot five-link dynamics and state-phased virtual-constraint tracking.
//! See docs/rabbit-model.md for coordinates, derivation and validity limits.
use crate::SimulationConfig;
use nalgebra::{SMatrix, SVector};
use serde::Deserialize;
type V5 = SVector<f64, 5>;
type V4 = SVector<f64, 4>;
type M5 = SMatrix<f64, 5, 5>;
type J4 = SMatrix<f64, 4, 5>;

#[derive(Clone, Debug, Deserialize)]
pub struct RabbitTrajectory {
    pub schema_version: u32,
    pub step_length_m: f64,
    pub speed_mps: f64,
    pub duration_s: f64,
    pub phase: Vec<f64>,
    pub q: Vec<[f64; 5]>,
    pub dq: Vec<[f64; 5]>,
    pub torque: Vec<[f64; 4]>,
    pub max_constraint_violation: f64,
}

impl RabbitTrajectory {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let n = p.phase.len();
        if p.schema_version != 1
            || n < 3
            || p.q.len() != n
            || p.dq.len() != n
            || p.torque.len() != n
            || !p.duration_s.is_finite()
            || p.duration_s <= 0.0
            || !p.step_length_m.is_finite()
            || p.step_length_m <= 0.0
            || !p.speed_mps.is_finite()
            || p.speed_mps <= 0.0
            || !p.max_constraint_violation.is_finite()
            || p.max_constraint_violation < 0.0
            || p.max_constraint_violation > 1e-4
            || p.phase.iter().any(|v| !v.is_finite())
            || p.q
                .iter()
                .flatten()
                .chain(p.dq.iter().flatten())
                .chain(p.torque.iter().flatten())
                .any(|v| !v.is_finite())
            || p.phase.windows(2).any(|w| w[1] <= w[0])
            || p.phase[0].abs() > 1e-9
            || (p.phase[n - 1] - 1.0).abs() > 1e-9
        {
            return Err("invalid or infeasible RABBIT trajectory".into());
        }
        Ok(p)
    }
    pub fn blend(&self, other: &Self, weight: f64) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let mut p = self.clone();
        p.duration_s = self.duration_s * (1.0 - w) + other.duration_s * w;
        p.speed_mps = self.speed_mps * (1.0 - w) + other.speed_mps * w;
        p.step_length_m = self.step_length_m * (1.0 - w) + other.step_length_m * w;
        for i in 0..p.phase.len() {
            let (a, va, ua) = self.sample(p.phase[i]);
            let (b, vb, ub) = other.sample(p.phase[i]);
            p.q[i] = (a * (1.0 - w) + b * w).into();
            p.dq[i] = ((va * self.duration_s * (1.0 - w) + vb * other.duration_s * w)
                / p.duration_s)
                .into();
            p.torque[i] = (ua * (1.0 - w) + ub * w).into();
        }
        p
    }
    pub fn sample(&self, s: f64) -> (V5, V5, V4) {
        let s = s.clamp(0.0, 1.0);
        let i = self
            .phase
            .partition_point(|x| *x <= s)
            .saturating_sub(1)
            .min(self.phase.len() - 2);
        let ds = self.phase[i + 1] - self.phase[i];
        let t = (s - self.phase[i]) / ds;
        let h = ds * self.duration_s;
        let q0 = V5::from_row_slice(&self.q[i]);
        let q1 = V5::from_row_slice(&self.q[i + 1]);
        let v0 = V5::from_row_slice(&self.dq[i]);
        let v1 = V5::from_row_slice(&self.dq[i + 1]);
        let q = (2.0 * t.powi(3) - 3.0 * t * t + 1.0) * q0
            + (t.powi(3) - 2.0 * t * t + t) * h * v0
            + (-2.0 * t.powi(3) + 3.0 * t * t) * q1
            + (t.powi(3) - t * t) * h * v1;
        let v = ((6.0 * t * t - 6.0 * t) * q0
            + (3.0 * t * t - 4.0 * t + 1.0) * h * v0
            + (-6.0 * t * t + 6.0 * t) * q1
            + (3.0 * t * t - 2.0 * t) * h * v1)
            / h;
        let u = V4::from_row_slice(&self.torque[i]) * (1.0 - t)
            + V4::from_row_slice(&self.torque[i + 1]) * t;
        (q, v, u)
    }
    pub fn sample_kinematics(&self, s: f64) -> (V5, V5, V5) {
        let s = s.clamp(0.0, 1.0);
        let i = self
            .phase
            .partition_point(|x| *x <= s)
            .saturating_sub(1)
            .min(self.phase.len() - 2);
        let ds = self.phase[i + 1] - self.phase[i];
        let t = (s - self.phase[i]) / ds;
        let h = ds * self.duration_s;
        let q0 = V5::from_row_slice(&self.q[i]);
        let q1 = V5::from_row_slice(&self.q[i + 1]);
        let v0 = V5::from_row_slice(&self.dq[i]);
        let v1 = V5::from_row_slice(&self.dq[i + 1]);
        let acceleration = ((12.0 * t - 6.0) * q0
            + (6.0 * t - 4.0) * h * v0
            + (-12.0 * t + 6.0) * q1
            + (6.0 * t - 2.0) * h * v1)
            / (h * h);
        let (q, v, _) = self.sample(s);
        (q, v, acceleration)
    }
}

#[derive(Clone)]
pub struct RabbitModel {
    h: M5,
    inertia: V5,
    gravity: V5,
    linear_damping: f64,
    angular_damping: f64,
    shin: f64,
    thigh: f64,
    torque_limit: f64,
    velocity_limit: f64,
    pub b: SMatrix<f64, 5, 4>,
}
impl RabbitModel {
    pub fn new(c: &SimulationConfig) -> Self {
        let r = &c.robot;
        let ls = r.shin.length as f64;
        let lt = r.thigh.length as f64;
        let lb = r.torso.length as f64;
        let links = [&r.shin, &r.thigh, &r.torso, &r.thigh, &r.shin];
        let a = M5::from_row_slice(&[
            ls / 2.0,
            0.,
            0.,
            0.,
            0.,
            ls,
            lt / 2.0,
            0.,
            0.,
            0.,
            ls,
            lt,
            lb / 2.0,
            0.,
            0.,
            ls,
            lt,
            0.,
            -lt / 2.0,
            0.,
            ls,
            lt,
            0.,
            -lt,
            -ls / 2.0,
        ]);
        let mass = V5::from_fn(|i, _| links[i].mass as f64);
        let inertia = V5::from_fn(|i, _| {
            links[i].mass as f64
                * ((links[i].length as f64).powi(2) + (links[i].width as f64).powi(2))
                / 12.0
        });
        let h = a.transpose() * M5::from_diagonal(&mass) * a;
        let gravity = a.transpose() * mass * (c.physics.gravity_y.abs() as f64);
        Self {
            h,
            inertia,
            gravity,
            linear_damping: r.body_dynamics.linear_damping as f64,
            angular_damping: r.body_dynamics.angular_damping as f64,
            shin: ls,
            thigh: lt,
            torque_limit: c.servo.max_torque as f64,
            velocity_limit: (c.servo.max_speed_deg_s as f64).to_radians(),
            b: SMatrix::from_row_slice(&[
                0., 1., 0., 0., 1., -1., 0., 0., -1., 0., -1., 0., 0., 0., 1., -1., 0., 0., 0., 1.,
            ]),
        }
    }
    pub fn validate_trajectory(&self, p: &RabbitTrajectory) -> Result<(), String> {
        let mut last = self.hip_x(&p.sample(0.0).0);
        for i in 1..=1000 {
            let next = self.hip_x(&p.sample(i as f64 / 1000.0).0);
            if next <= last {
                return Err("RABBIT hip phase is not monotone".into());
            }
            let (q, v, a) = p.sample_kinematics(i as f64 / 1000.0);
            let u = p.sample(i as f64 / 1000.0).2;
            if u.amax() > self.torque_limit + 1e-5
                || (self.b.transpose() * v).amax() > self.velocity_limit + 1e-5
            {
                return Err("trajectory exceeds current motor limits".into());
            }
            if (self.mass(&q) * a + self.bias(&q, &v) - self.b * u).amax() > 0.02 {
                return Err("trajectory does not satisfy this robot's dynamics".into());
            }
            last = next;
        }
        Ok(())
    }
    pub fn mass(&self, q: &V5) -> M5 {
        M5::from_fn(|i, j| {
            self.h[(i, j)] * (q[i] - q[j]).cos() + if i == j { self.inertia[i] } else { 0. }
        })
    }
    pub fn bias(&self, q: &V5, v: &V5) -> V5 {
        V5::from_fn(|i, _| {
            let coriolis = (0..5)
                .map(|j| self.h[(i, j)] * (q[i] - q[j]).sin() * v[j] * v[j])
                .sum::<f64>();
            let damping = (0..5)
                .map(|j| self.linear_damping * self.h[(i, j)] * (q[i] - q[j]).cos() * v[j])
                .sum::<f64>()
                + self.angular_damping * self.inertia[i] * v[i];
            coriolis - self.gravity[i] * q[i].sin() + damping
        })
    }
    fn hip_x(&self, q: &V5) -> f64 {
        -self.shin * q[0].sin() - self.thigh * q[1].sin()
    }
    pub fn phase(&self, q: &V5, p: &RabbitTrajectory) -> f64 {
        let x = self.hip_x(q);
        // Interpolate the monotone hip coordinate of the optimized orbit.
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..30 {
            let mid = (lo + hi) * 0.5;
            if self.hip_x(&p.sample(mid).0) < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) * 0.5
    }
    /// State phase is based on hip progress, so faster motion advances motors faster.
    /// Returns None for a non-monotone or singular reference.
    pub fn control(&self, q: &V5, v: &V5, p: &RabbitTrajectory, kp: f64, kd: f64) -> Option<V4> {
        let phase = self.phase(q, p);
        let (qr, vr, ar) = p.sample_kinematics(phase);
        let grad = V5::new(
            -self.shin * q[0].cos(),
            -self.thigh * q[1].cos(),
            0.,
            0.,
            0.,
        );
        let ref_grad = V5::new(
            -self.shin * qr[0].cos(),
            -self.thigh * qr[1].cos(),
            0.,
            0.,
            0.,
        );
        let speed = ref_grad.dot(&vr);
        if speed <= 1e-5 {
            return None;
        }
        let ratio = grad.dot(v) / speed;
        let hip_bias = self.shin * q[0].sin() * v[0] * v[0] + self.thigh * q[1].sin() * v[1] * v[1];
        let ref_accel = ref_grad.dot(&ar)
            + self.shin * qr[0].sin() * vr[0] * vr[0]
            + self.thigh * qr[1].sin() * vr[1] * vr[1];
        let j: J4 = self.b.transpose() * (M5::identity() - vr * grad.transpose() / speed);
        let y = self.b.transpose() * (q - qr);
        let acceleration_bias = -self.b.transpose()
            * (ar * ratio * ratio + vr * (hip_bias - ref_accel * ratio * ratio) / speed);
        let m = self.mass(q).lu();
        let minvb = m.solve(&self.b)?;
        let minvh = m.solve(&self.bias(q, v))?;
        let a = j * minvb;
        let rhs = -kp * y - kd * (j * v) - acceleration_bias + j * minvh;
        let torque = a.lu().solve(&rhs)?;
        torque.iter().all(|x| x.is_finite()).then_some(torque)
    }
}
