import unittest
from pathlib import Path
import numpy as np
from model import FiveLinkModel

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=FiveLinkModel(Path(__file__).resolve().parents[1]/'robot_config.toml')

    def test_mass_and_internal_actuation(self):
        rng=np.random.default_rng(71)
        for _ in range(50):
            q=rng.uniform(-.6,.6,5);m=np.array(self.model.mass(q))
            np.testing.assert_allclose(m,m.T,atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(m).min(),0)
        np.testing.assert_array_equal(self.model.B.sum(axis=0),0)
        self.assertEqual(np.linalg.matrix_rank(self.model.B),4)

    def test_inverse_dynamics_recovers_torque_and_exposes_unactuated_residual(self):
        q=np.array([-.1,.2,-.05,.3,-.2]);v=np.array([-.3,.2,.1,.4,-.1]);u=np.array([.2,-.1,.3,.1])
        acceleration=np.array(self.model.dynamics(q,v,u)).ravel()
        recovered,residual=self.model.inverse_dynamics(q,v,acceleration)
        np.testing.assert_allclose(recovered,u,atol=1e-11)
        np.testing.assert_allclose(residual,0,atol=1e-11)
        _,infeasible=self.model.inverse_dynamics(q,v,np.zeros(5))
        self.assertGreater(np.linalg.norm(infeasible),.01)

    def test_impact_does_not_add_energy(self):
        rng=np.random.default_rng(18)
        # Reflection-symmetric configuration has both endpoints on the floor.
        for _ in range(30):
            q=np.array([-.2,.15,-.05,-.15,.2]);v=rng.normal(0,.4,5)
            after,_=self.model.impact(q,v)
            after=np.array(after).ravel();qp=q[self.model.permutation]
            before_ke=.5*v@np.array(self.model.mass(q))@v
            after_ke=.5*after@np.array(self.model.mass(qp))@after
            self.assertLessEqual(after_ke,before_ke+1e-10)

    def test_new_contact_has_zero_velocity_after_impact(self):
        q=np.array([-.2,.15,-.05,-.15,.2]);v=np.array([.3,-.2,.1,.4,-.5])
        np.testing.assert_allclose(self.model.impact_contact_velocity(q,v),0,atol=1e-12)

    def test_passive_energy_loss_equals_damping_power(self):
        q=np.array([-.1,.2,-.05,.3,-.2]);v=np.array([-.3,.2,.1,.4,-.1])
        a=np.array(self.model.dynamics(q,v,np.zeros(4))).ravel();eps=1e-6
        derivative=(float(self.model.energy(q+eps*v,v+eps*a))-float(self.model.energy(q-eps*v,v-eps*a)))/(2*eps)
        self.assertAlmostEqual(derivative,-float(self.model.damping_power(q,v)),places=7)

    def test_forward_geometry_and_joint_signs(self):
        hip,foot,com,points,joints=self.model.geometry(np.zeros(5))
        np.testing.assert_allclose(np.array(hip).ravel(),[0,.96],atol=1e-7)
        np.testing.assert_allclose(np.array(foot).ravel(),[0,0],atol=1e-12)
        np.testing.assert_allclose(joints,0)

if __name__=='__main__':unittest.main()
