"""Numerical nine-dimensional post-impact return map (point-foot model only)."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import FiveLinkModel
from closed_loop import Controller

def analyze(config,trajectory):
    m=FiveLinkModel(config);p=json.loads(Path(trajectory).read_text());c=Controller(m,p)
    ls,lt,*_=m.lengths;limit=m.config['servo']['max_torque'];nominal=np.r_[p['q'][0][:4],p['dq'][0]]
    sign=np.sign(p['q'][0][4])
    def expand(z):
        a=(ls*np.cos(z[0])+lt*np.cos(z[1])-lt*np.cos(z[3]))/ls
        if abs(a)>=1:raise ValueError('perturbation outside the touchdown section')
        return np.r_[z[:4],sign*np.arccos(a),z[4:]]
    def mapping(z):
        def ode(t,y):return np.r_[y[5:],np.array(m.dynamics(y[:5],y[5:],np.clip(c.control(y[:5],y[5:]),-limit,limit))).ravel()]
        def contact(t,y):return .01 if t<.05 else float(m.geometry(y[:5])[1][1])
        contact.direction=-1;contact.terminal=True
        sol=solve_ivp(ode,[0,2*p['duration_s']],expand(z),events=contact,rtol=1e-9,atol=1e-11,max_step=p['duration_s']/400)
        if not sol.success or not len(sol.t_events[0]):raise RuntimeError('return map has no touchdown')
        q=sol.y[:5,-1];v=sol.y[5:,-1];vp,_=m.impact(q,v);qp=q[m.permutation]
        return np.r_[qp[:4],np.array(vp).ravel()]
    fixed=nominal.copy()
    for _ in range(12):fixed=mapping(fixed)
    residual=float(np.linalg.norm(mapping(fixed)-fixed))
    matrices=[];radii=[]
    for epsilon in [1e-5,5e-6]:
        jac=np.column_stack([(mapping(fixed+np.eye(9)[i]*epsilon)-mapping(fixed-np.eye(9)[i]*epsilon))/(2*epsilon) for i in range(9)])
        matrices.append(jac);radii.append(float(max(abs(np.linalg.eigvals(jac)))))
    difference=float(np.max(np.abs(matrices[1]-matrices[0])))
    report=dict(model='point-foot, saturated state-phased feedback; not Rapier',section='post-impact old-foot height zero, q[0:4] and dq[0:5]',fixed_point_residual=residual,spectral_radii=radii,jacobian_refinement_difference=difference,jacobian=matrices[-1].tolist(),numerically_locally_stable=bool(residual<1e-4 and difference<.05 and max(radii)<1))
    Path(trajectory).with_suffix('.poincare.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='jacobian'},indent=2))
    if not report['numerically_locally_stable']:raise RuntimeError('local numerical stability check did not pass')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--config',default='robot_config.toml');a=p.parse_args();analyze(a.config,a.trajectory)
