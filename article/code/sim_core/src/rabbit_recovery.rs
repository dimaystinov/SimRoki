//! Recovery uses joint motor targets only. No pose reset or external torso force.
use serde::{Deserialize, Serialize};

/// Learned, independent joint targets with elapsed-time input. A low torso no
/// longer forces the controller to hold the same crouch indefinitely.
#[derive(Clone, Deserialize, Serialize)]
pub struct RecoveryMotion {
    #[serde(default)]
    pub start_after_s: f64,
    pub phase_s: Vec<f64>,
    pub joint_target_rad: Vec<[f64;4]>,
    pub pitch_gain: f64,
    pub omega_gain: f64,
}

impl RecoveryMotion {
    pub fn validate(&self)->Result<(),String> {
        if !self.start_after_s.is_finite() || !(0.0..=10.0).contains(&self.start_after_s)
            || self.phase_s.len()<2 || self.phase_s.len()>64 || self.phase_s[0]!=0.0
            || self.phase_s.iter().any(|x|!x.is_finite() || *x<0.0 || *x>30.0)
            || self.phase_s.windows(2).any(|w|w[0]>=w[1])
            || self.joint_target_rad.len()!=self.phase_s.len()
            || self.joint_target_rad.iter().flatten().any(|x|!x.is_finite() || x.abs()>2.8)
            || !self.pitch_gain.is_finite() || self.pitch_gain.abs()>3.0
            || !self.omega_gain.is_finite() || self.omega_gain.abs()>1.0 {
            return Err("invalid recovery motion phases, motor targets or feedback gains".into());
        }
        Ok(())
    }

    pub fn sample(&self,time:f64,pitch:f64,omega:f64)->[f64;4] {
        let t=time.clamp(0.0,*self.phase_s.last().unwrap());
        let i=self.phase_s.partition_point(|v|*v<=t).saturating_sub(1).min(self.phase_s.len()-2);
        let a=(t-self.phase_s[i])/(self.phase_s[i+1]-self.phase_s[i]);
        std::array::from_fn(|j| {
            let feedback=if j%2==0 {self.pitch_gain*pitch+self.omega_gain*omega}else{0.0};
            ((1.0-a)*self.joint_target_rad[i][j]+a*self.joint_target_rad[i+1][j]+feedback).clamp(-2.8,2.8)
        })
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct RecoveryPolicy {
    pub height_m: Vec<f64>,
    pub pitch_rad: Vec<f64>,
    pub angular_speed_rad_s: Vec<f64>,
    pub hip_target_rad: Vec<Vec<Vec<f64>>>,
    pub knee_target_rad: Vec<Vec<Vec<f64>>>,
}

impl RecoveryPolicy {
    pub fn validate(&self)->Result<(),String> {
        for axis in [&self.height_m,&self.pitch_rad,&self.angular_speed_rad_s] {
            if axis.len()<2 || axis.len()>64 || axis.iter().any(|v|!v.is_finite()) || axis.windows(2).any(|w|w[0]>=w[1]) {
                return Err("recovery axes must be finite, strictly increasing, length 2..64".into());
            }
        }
        for values in [&self.hip_target_rad,&self.knee_target_rad] {
            if values.len()!=self.height_m.len() || values.iter().any(|p|p.len()!=self.pitch_rad.len() || p.iter().any(|r|r.len()!=self.angular_speed_rad_s.len() || r.iter().any(|v|!v.is_finite() || v.abs()>2.8))) {
                return Err("invalid recovery motor target map".into());
            }
        }
        Ok(())
    }

    pub fn sample(&self,height:f64,pitch:f64,omega:f64)->[f64;2] {
        let locate=|axis:&[f64],x:f64| {
            let x=x.clamp(axis[0],axis[axis.len()-1]);
            let i=axis.partition_point(|v|*v<=x).saturating_sub(1).min(axis.len()-2);
            (i,(x-axis[i])/(axis[i+1]-axis[i]))
        };
        let (i,a)=locate(&self.height_m,height);
        let (j,b)=locate(&self.pitch_rad,pitch);
        let (k,c)=locate(&self.angular_speed_rad_s,omega);
        let sample=|v:&Vec<Vec<Vec<f64>>>| {
            let mut value=0.0;
            for di in 0..2 {for dj in 0..2 {for dk in 0..2 {
                value+=v[i+di][j+dj][k+dk]*if di==0 {1.0-a}else{a}*if dj==0 {1.0-b}else{b}*if dk==0 {1.0-c}else{c};
            }}}
            value
        };
        [sample(&self.hip_target_rad),sample(&self.knee_target_rad)]
    }
}

#[derive(Clone, Copy, Default, Serialize)]
pub struct RecoveryStatus {
    pub active: bool,
    pub elapsed_s: f32,
    pub stable_s: f32,
    pub fall_events: u32,
    pub recoveries: u32,
    pub exhausted: bool,
}

/// Classical seed teacher; learned maps can replace both motor targets.
pub fn seed_targets(height:f64,pitch:f64,omega:f64)->[f64;2] {
    let bend=(0.25+1.45*((1.15-height)/0.8).clamp(0.0,1.0)).clamp(0.2,1.8);
    [(0.5*bend+1.3*pitch+0.22*omega).clamp(-2.6,2.6),-bend]
}
