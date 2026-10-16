"""Independent hybrid simulation of state-phased virtual constraints."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.integrate import solve_ivp
from model import FiveLinkModel

class Controller:
    def __init__(self,model,plan):
        self.m=model;self.p=plan;self.T=plan['duration_s'];self.ls,self.lt,*_=model.lengths
        self.path=CubicHermiteSpline(np.array(plan['phase'])*self.T,plan['q'],plan['dq'])
        self.ts=np.linspace(0,self.T,2001);qq=self.path(self.ts)
        self.xs=-self.ls*np.sin(qq[:,0])-self.lt*np.sin(qq[:,1])
        if np.min(np.diff(self.xs))<=0:raise ValueError('hip phase must be monotone')
    def control(self,q,v):
        ls,lt=self.ls,self.lt
        x=-ls*np.sin(q[0])-lt*np.sin(q[1]);t=np.interp(x,self.xs,self.ts)
        qr=self.path(t);vr=self.path(t,1);ar=self.path(t,2)
        grad=np.array([-ls*np.cos(q[0]),-lt*np.cos(q[1]),0,0,0])
        gr=np.array([-ls*np.cos(qr[0]),-lt*np.cos(qr[1]),0,0,0]);speed=gr@vr
        ratio=grad@v/speed
        hb=ls*np.sin(q[0])*v[0]**2+lt*np.sin(q[1])*v[1]**2
        ra=gr@ar+ls*np.sin(qr[0])*vr[0]**2+lt*np.sin(qr[1])*vr[1]**2
        J=self.m.B.T@(np.eye(5)-np.outer(vr,grad)/speed)
        bias=-self.m.B.T@(ar*ratio**2+vr*(hb-ra*ratio**2)/speed)
        M=np.array(self.m.mass(q));H=np.array(self.m.bias(q,v)).ravel()
        A=J@np.linalg.solve(M,self.m.B)
        rhs=-100*self.m.B.T@(q-qr)-20*J@v-bias+J@np.linalg.solve(M,H)
        return np.linalg.solve(A,rhs)

def run(config,trajectory,steps=20,perturb=0.):
    m=FiveLinkModel(config);p=json.loads(Path(trajectory).read_text());c=Controller(m,p)
    state=np.r_[p['q'][0],p['dq'][0]];state[2]+=perturb;rows=[];limit=m.config['servo']['max_torque']
    for step in range(steps):
        peak=[0.]
        def ode(t,z):
            u=c.control(z[:5],z[5:]);peak[0]=max(peak[0],float(np.max(np.abs(u))))
            return np.r_[z[5:],np.array(m.dynamics(z[:5],z[5:],np.clip(u,-limit,limit))).ravel()]
        def landing(t,z):
            if t<.05:return .01
            return float(m.geometry(z[:5])[1][1])
        landing.terminal=True;landing.direction=-1
        sol=solve_ivp(ode,[0,2*p['duration_s']],state,events=landing,rtol=1e-7,atol=1e-9,max_step=p['duration_s']/150)
        if not sol.success or not len(sol.t_events[0]):raise RuntimeError(f'no touchdown at step {step}')
        q=sol.y[:5,-1];v=sol.y[5:,-1];distance=float(m.geometry(q)[1][0]);vp,_=m.impact(q,v)
        state=np.r_[q[m.permutation],np.array(vp).ravel()]
        rows.append(dict(step=step,length_m=distance,duration_s=float(sol.t[-1]),peak_requested_torque_nm=peak[0],return_error=float(np.linalg.norm(state-np.r_[p['q'][0],p['dq'][0]]))))
        if distance<=0 or np.max(np.abs(q))>1.2:raise RuntimeError(f'fell at step {step}: {rows[-1]}')
    report=dict(steps=rows,perturbation_rad=perturb,model='point-foot; not Rapier contact validation')
    out=Path(trajectory).with_suffix('.closed-loop.json');out.write_text(json.dumps(report,indent=2));print(json.dumps(rows[-1],indent=2));return report
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('trajectory');a.add_argument('--config',default='robot_config.toml');a.add_argument('--steps',type=int,default=20);a.add_argument('--perturb',type=float,default=0.);v=a.parse_args();run(v.config,v.trajectory,v.steps,v.perturb)
