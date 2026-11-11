"""Bounded covariance CEM with antithetic samples and deterministic replay."""
import math
import numpy as np


BOUNDS=np.array([(.12,.26),(0,.07),(0,.3),(-.08,.2),(-.01,.03),(0,1),(-1,1),(-2,2),(-.25,.25),(.4,2),(.4,2),
                 (.8,2.6),(.05,.8),(-.5,3),(-.3,1)],dtype=float)
SCALES=np.array([.007,.006,.02,.02,.002,.2,.2,.3,.04,.12,.12,.25,.1,.3,.1])


class CovarianceSearch:
    def __init__(self,center,state=None):
        self.low=BOUNDS[:,0];self.span=BOUNDS[:,1]-self.low
        self.mean=(np.asarray(center)-self.low)/self.span
        self.floor=(SCALES/self.span*.15)**2
        self.cov=np.diag((SCALES/self.span)**2)
        self.stagnation=0
        if state:
            self.mean=np.asarray(state['mean']);self.cov=np.asarray(state['cov']);self.stagnation=state['stagnation']

    def sample(self,population,seed,incumbent):
        rng=np.random.default_rng(seed)
        values=[np.asarray(incumbent),self.low+np.clip(self.mean,0,1)*self.span]
        cov=self.cov*(1+min(self.stagnation,20)*.12)
        while len(values)<population:
            noise=rng.multivariate_normal(np.zeros(len(self.mean)),cov)
            for sign in (1,-1):
                values.append(self.low+np.clip(self.mean+sign*noise,0,1)*self.span)
                if len(values)==population:break
        if population>=8:
            # A broad exploratory sample prevents permanent collapse around an early elite.
            values[-1]=self.low+np.clip(self.mean+rng.normal(0,SCALES/self.span*2),0,1)*self.span
        return [v.tolist() for v in values]

    def update(self,candidates,improved):
        ordered=sorted(candidates,key=lambda c:(c['training'].get('finishers',0),
            -c['training'].get('fall_events',0),c['training']['score']),reverse=True)
        elite=ordered[:max(2,math.ceil(len(ordered)*.25))]
        x=np.array([(np.asarray(c['parameters'])-self.low)/self.span for c in elite])
        weights=np.log(len(elite)+.5)-np.log(np.arange(1,len(elite)+1));weights/=weights.sum()
        mean=np.sum(x*weights[:,None],axis=0)
        delta=x-mean
        covariance=(delta*weights[:,None]).T@delta
        self.mean=.7*self.mean+.3*mean
        self.cov=.7*self.cov+.3*covariance
        # Keep covariance positive definite and every coordinate exploratory.
        self.cov=(self.cov+self.cov.T)*.5+np.diag(self.floor)
        self.stagnation=0 if improved else self.stagnation+1

    def state(self):
        return {'mean':self.mean.tolist(),'cov':self.cov.tolist(),'stagnation':self.stagnation}
