"""Five-link planar model in SimRoki's world-angle convention.

Independent Lagrangian derivation; morphology and hybrid construction follow
RABBIT (Westervelt et al., 2003). Parameters come from robot_config.toml.
Coordinates: stance shin/thigh, torso, swing thigh/shin. No ankle actuator.
"""
from pathlib import Path
import tomllib
import casadi as ca
import numpy as np


class FiveLinkModel:
    def __init__(self, config_path):
        self.config = tomllib.loads(Path(config_path).read_text())
        r = self.config['robot']
        self.lengths = [r[k]['length'] for k in ['shin','thigh','torso','thigh','shin']]
        self.masses = np.array([r[k]['mass'] for k in ['shin','thigh','torso','thigh','shin']])
        self.inertias = np.array([r[k]['mass']*(r[k]['length']**2+r[k]['width']**2)/12
                                  for k in ['shin','thigh','torso','thigh','shin']])
        self.gravity = abs(self.config['physics']['gravity_y'])
        self.permutation = [4,3,2,1,0]
        q = ca.SX.sym('q',5); dq = ca.SX.sym('dq',5); u = ca.SX.sym('u',4)
        ls, lt, lb, _, _ = self.lengths
        def down(a, l): return ca.vertcat(l*ca.sin(a), -l*ca.cos(a))
        origin = ca.SX.zeros(2)
        knee = -down(q[0],ls)
        hip = knee-down(q[1],lt)
        swing_knee = hip+down(q[3],lt)
        foot = swing_knee+down(q[4],ls)
        centers = [knee/2, (knee+hip)/2, hip-down(q[2],lb/2),
                   (hip+swing_knee)/2, (swing_knee+foot)/2]
        points = ca.horzcat(origin,knee,hip,hip-down(q[2],lb),swing_knee,foot)
        M = ca.diag(ca.DM(self.inertias)); V=0; Mt=ca.SX.zeros(5,5)
        Jcom = ca.SX.zeros(2,5)
        for mass,p in zip(self.masses,centers):
            J=ca.jacobian(p,q); Mt+=mass*J.T@J; V+=mass*self.gravity*p[1]
            Jcom+=mass*J/sum(self.masses)
        M+=Mt
        kinetic=0.5*ca.mtimes([dq.T,M,dq])
        bias=ca.jacobian(ca.gradient(kinetic,dq),q)@dq-ca.gradient(kinetic,q)+ca.gradient(V,q)
        damp=r['body_dynamics']
        damping=damp['linear_damping']*Mt+damp['angular_damping']*ca.diag(ca.DM(self.inertias))
        B=ca.DM([[0,1,0,0],[1,-1,0,0],[-1,0,-1,0],[0,0,1,-1],[0,0,0,1]])
        ddq=ca.solve(M,B@u-bias-damping@dq)
        com=sum(m*p for m,p in zip(self.masses,centers))/sum(self.masses)
        com_acc=Jcom@ddq+ca.jacobian(Jcom@dq,q)@dq
        reaction=sum(self.masses)*(com_acc+ca.vertcat(0,self.gravity))
        # Extend to a floating base for the inelastic new-foot impact.
        Me=ca.SX.zeros(7,7); Me[:5,:5]=ca.diag(ca.DM(self.inertias))
        for mass,p in zip(self.masses,centers):
            J=ca.horzcat(ca.jacobian(p,q),ca.SX.eye(2));Me+=mass*J.T@J
        Jnew=ca.horzcat(ca.jacobian(foot,q),ca.SX.eye(2))
        before=ca.vertcat(dq,0,0)
        impulse=-ca.solve(Jnew@ca.solve(Me,Jnew.T),Jnew@before)
        after=before+ca.solve(Me,Jnew.T@impulse)
        joints=ca.vertcat(q[1]-q[2],q[0]-q[1],q[3]-q[2],q[4]-q[3])
        self.mass=ca.Function('mass',[q],[M])
        self.bias=ca.Function('bias',[q,dq],[bias+damping@dq])
        self.energy=ca.Function('energy',[q,dq],[kinetic+V])
        self.dynamics=ca.Function('dynamics',[q,dq,u],[ddq])
        self.reaction=ca.Function('reaction',[q,dq,u],[reaction])
        self.impact=ca.Function('impact',[q,dq],[after[:5][self.permutation],impulse])
        self.impact_contact_velocity=ca.Function('impact_contact_velocity',[q,dq],[Jnew@after])
        self.damping_power=ca.Function('damping_power',[q,dq],[ca.mtimes([dq.T,damping,dq])])
        self.geometry=ca.Function('geometry',[q],[hip,foot,com,points,joints])
        self.foot_velocity=ca.Function('foot_velocity',[q,dq],[ca.jacobian(foot,q)@dq])
        self.joint_velocity=ca.Function('joint_velocity',[dq],[ca.jacobian(joints,q)@dq])
        self.B=np.array(B)

    def inverse_dynamics(self,q,dq,ddq):
        """Actuated torques plus the unactuated residual; never hide infeasibility."""
        rhs=np.array(self.mass(q))@np.asarray(ddq)+np.array(self.bias(q,dq)).ravel()
        torque=np.linalg.lstsq(self.B,rhs,rcond=None)[0]
        return torque,rhs-self.B@torque
