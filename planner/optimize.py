"""Constrained periodic gait via Hermite-Simpson direct collocation.

This solves a point-foot model. A solution is not a proof of closed-loop stability
or of feasibility under Rapier's finite-width contact geometry.
"""
import argparse,json,hashlib
from pathlib import Path
import casadi as ca
import numpy as np
from model import FiveLinkModel


def solve(config, output, length=.20, speed=.3, intervals=40, initial=None, clearance=.025, lean=.06, constraint_margin=0.0):
    model=FiveLinkModel(config); op=ca.Opti(); n=intervals
    Q=op.variable(5,n+1); V=op.variable(5,n+1); U=op.variable(4,n+1)
    T=length/speed; dt=T/n
    if not 0 <= constraint_margin < .25:raise ValueError('constraint margin must be in [0, .25)')
    reserve=1-constraint_margin
    limit=model.config['servo']['max_torque']*reserve
    vmax=np.deg2rad(model.config['servo']['max_speed_deg_s'])*reserve
    mu=min(model.config['physics']['ground_friction'],model.config['robot']['shin']['friction'])*reserve
    op.subject_to(op.bounded(-1.2,Q,1.2))
    op.subject_to(op.bounded(-vmax,V,vmax))
    op.subject_to(op.bounded(-limit,U,limit))
    cost=0
    def constrain(q,v,u,phase):
        hip,foot,com,points,joints=model.geometry(q)
        force=model.reaction(q,v,u)
        op.subject_to(op.bounded(-.28,q[2],.05))
        op.subject_to(op.bounded(.75,hip[1],.96))
        op.subject_to(op.bounded(-1.3,joints[1],-.015))
        op.subject_to(op.bounded(-1.3,joints[3],-.015))
        op.subject_to(op.bounded(-vmax,model.joint_velocity(v),vmax))
        op.subject_to(force[1]>=.05)
        op.subject_to(-model.lengths[0]*ca.cos(q[0])*v[0]-model.lengths[1]*ca.cos(q[1])*v[1]>=.01)
        op.subject_to(op.bounded(-mu*force[1],force[0],mu*force[1]))
        op.subject_to(foot[1]>=clearance*np.sin(np.pi*phase)**2)
        op.subject_to(foot[1]<=.15)
        op.subject_to(points[1,1:5]>=.02)
    for i in range(n+1):
        constrain(Q[:,i],V[:,i],U[:,i],i/n)
        cost+=dt*(ca.sumsqr(U[:,i])+.05*ca.sumsqr(V[:,i]) + 2*(Q[2,i]+lean)**2)
    for i in range(n):
        q0,q1=Q[:,i],Q[:,i+1];v0,v1=V[:,i],V[:,i+1];u0,u1=U[:,i],U[:,i+1]
        a0=model.dynamics(q0,v0,u0);a1=model.dynamics(q1,v1,u1)
        qm=(q0+q1)/2+dt*(v0-v1)/8;vm=(v0+v1)/2+dt*(a0-a1)/8;um=(u0+u1)/2
        am=model.dynamics(qm,vm,um)
        op.subject_to(q1-q0==dt*(v0+4*vm+v1)/6)
        op.subject_to(v1-v0==dt*(a0+4*am+a1)/6)
        constrain(qm,vm,um,(i+.5)/n)
        cost+=.001*ca.sumsqr(u1-u0)
    op.subject_to(Q[:,0]==Q[model.permutation,-1])
    impact,impulse=model.impact(Q[:,-1],V[:,-1])
    op.subject_to(V[:,0]==impact)
    op.subject_to(impulse[1]>=0)
    op.subject_to(op.bounded(-mu*impulse[1],impulse[0],mu*impulse[1]))
    hip0,foot0,*_=model.geometry(Q[:,0]);hip1,foot1,*_=model.geometry(Q[:,-1])
    op.subject_to(foot1==ca.vertcat(length,0))
    op.subject_to(model.foot_velocity(Q[:,-1],V[:,-1])[1]<=-.01)
    op.subject_to(hip1[0]>hip0[0])
    op.minimize(cost)
    if initial:
        old=json.loads(Path(initial).read_text()); t=np.linspace(0,1,n+1)
        for variable,key in [(Q,'q'),(V,'dq'),(U,'torque')]:
            a=np.array(old[key]);op.set_initial(variable,np.array([np.interp(t,old['phase'],row) for row in a.T]))
    else:
        # Bent-knee geometric seed; dynamics, contact and impact are enforced by NLP.
        t=np.linspace(0,1,n+1); q=np.zeros((5,n+1));hip_y=.91
        ls,lt,*_=model.lengths
        for i,s in enumerate(t):
            hip_x=-length/2+length*s;foot_x=-length+2*length*s;foot_y=.05*np.sin(np.pi*s)**2
            def leg(dx,dy):
                knee=-np.arccos(np.clip((dx*dx+dy*dy-lt*lt-ls*ls)/(2*lt*ls),-1,1))
                thigh=np.arctan2(dx,dy)-np.arctan2(ls*np.sin(knee),lt+ls*np.cos(knee))
                return thigh,thigh+knee
            sth,ssh=leg(-hip_x,hip_y);wth,wsh=leg(foot_x-hip_x,hip_y-foot_y)
            q[:,i]=[ssh,sth,-.06,wth,wsh]
        op.set_initial(Q,q);op.set_initial(V,np.gradient(q,dt,axis=1));op.set_initial(U,0)
    op.solver('ipopt',{'expand':True},{'max_iter':2500,'tol':1e-7,'acceptable_tol':1e-5,'print_level':3})
    solution=op.solve()
    q=solution.value(Q);v=solution.value(V);u=solution.value(U)
    violation=max(float(np.max(np.maximum(solution.value(op.lbg)-solution.value(op.g),0))),
                  float(np.max(np.maximum(solution.value(op.g)-solution.value(op.ubg),0))))
    if not np.isfinite(violation) or violation>1e-4:raise RuntimeError(f'infeasible solution: {violation}')
    result={'schema_version':1,'model':'simroki-five-link-point-foot','config_sha256':hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            'step_length_m':length,'speed_mps':speed,'duration_s':T,'phase':np.linspace(0,1,n+1).tolist(),
            'q':q.T.tolist(),'dq':v.T.tolist(),'torque':u.T.tolist(),'max_constraint_violation':violation,
            'constraint_margin':constraint_margin,'minimum_clearance_m':clearance,'desired_forward_lean_rad':lean,'objective':float(solution.value(cost)),'solver_status':solution.stats()['return_status']}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['phase','q','dq','torque']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='robot_config.toml');p.add_argument('--output',default='motions/rabbit-20cm.json')
    p.add_argument('--step',type=float,default=.20);p.add_argument('--speed',type=float,default=.30);p.add_argument('--intervals',type=int,default=40);p.add_argument('--initial');p.add_argument('--clearance',type=float,default=.025);p.add_argument('--lean',type=float,default=.06);p.add_argument('--constraint-margin',type=float,default=0.0)
    a=p.parse_args();solve(a.config,a.output,a.step,a.speed,a.intervals,a.initial,a.clearance,a.lean,a.constraint_margin)
