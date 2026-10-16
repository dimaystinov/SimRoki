"""Independent dense and forward-integration checks for an optimized orbit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline,interp1d
from scipy.integrate import solve_ivp
from model import FiveLinkModel

def validate(config,trajectory):
    m=FiveLinkModel(config);p=json.loads(Path(trajectory).read_text());T=p['duration_s']
    t=np.array(p['phase'])*T;q=np.array(p['q']);v=np.array(p['dq']);u=np.array(p['torque'])
    path=CubicHermiteSpline(t,q,v);control=interp1d(t,u,axis=0,fill_value='extrapolate')
    dense=np.linspace(0,T,1001);max_residual=0;min_normal=1e9;min_clearance=1e9;friction_margin=1e9
    for x in dense:
        a=path(x,2);qq=path(x);vv=path(x,1);uu=control(x)
        residual=np.array(m.mass(qq))@a+np.array(m.bias(qq,vv)).ravel()-m.B@uu
        max_residual=max(max_residual,float(np.max(np.abs(residual))))
        force=np.array(m.reaction(qq,vv,uu)).ravel();min_normal=min(min_normal,float(force[1]))
        mu=min(m.config['physics']['ground_friction'],m.config['robot']['shin']['friction'])
        friction_margin=min(friction_margin,float(mu*force[1]-abs(force[0])))
        min_clearance=min(min_clearance,float(m.geometry(qq)[1][1]))
    def ode(x,state):return np.r_[state[5:],np.array(m.dynamics(state[:5],state[5:],control(x))).ravel()]
    sim=solve_ivp(ode,[0,T],np.r_[q[0],v[0]],rtol=1e-9,atol=1e-11,max_step=T/300)
    position_error=float(np.max(np.abs(sim.y[:5,-1]-q[-1])))
    velocity_error=float(np.max(np.abs(sim.y[5:,-1]-v[-1])))
    impact,_=m.impact(q[-1],v[-1]);periodic_error=float(np.max(np.abs(np.array(impact).ravel()-v[0])))
    report=dict(integration_success=sim.success,max_dense_dynamics_residual_nm=max_residual,
                min_normal_force_n=min_normal,min_friction_margin_n=friction_margin,
                min_swing_height_m=min_clearance,independent_position_error_rad=position_error,
                independent_velocity_error_rad_s=velocity_error,impact_periodicity_error=periodic_error,
                peak_torque_nm=float(np.max(np.abs(u))),step_length_m=p['step_length_m'],speed_mps=p['speed_mps'])
    print(json.dumps(report,indent=2));Path(trajectory).with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
    assert sim.success and max_residual<.02 and position_error<.01 and velocity_error<.03
    assert min_normal>0 and friction_margin>=-1e-5 and min_clearance>=-1e-5 and periodic_error<1e-5
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--config',default='robot_config.toml');a=p.parse_args();validate(a.config,a.trajectory)
