"""Explicit time-scaled costs, recoverable falls and risk-aware policy selection."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 2.0
    tracking: float = 2.0
    time_cost: float = .5
    wrong_contact: float = 40.0
    double_contact: float = 20.0
    fall_event: float = 150.0
    failure: float = 300.0
    timeout: float = 100.0
    finish: float = 200.0
    recovered: float = 50.0
    recovery_time: float = 2.0
    stall: float = 8.0
    work: float = .003
    action: float = .1
    action_change: float = .5


DEFAULT_REWARD=RewardConfig()


def step_reward(*,dt,dx,tracking,wrong_s,double_s,work_j,action_sq,action_change_sq,
                falls,recoveries,recovery_s,potential_change,stalled,reason,config=DEFAULT_REWARD):
    terms={
        'progress':config.progress*dx,
        'tracking':config.tracking*tracking*(dt-recovery_s),
        'time':-config.time_cost*dt,
        'wrong_contact':-config.wrong_contact*wrong_s,
        'double_contact':-config.double_contact*double_s,
        'work':-config.work*work_j,
        'action':-config.action*action_sq*dt,
        'action_change':-config.action_change*action_change_sq,
        'fall':-config.fall_event*falls,
        'recovered':config.recovered*recoveries,
        'recovery_time':-config.recovery_time*recovery_s,
        'recovery_progress':potential_change,
        'stall':-config.stall*dt*stalled,
        'terminal':config.finish if reason=='finish' else -config.failure if reason in ('fall','nonfinite') else -config.timeout if reason=='timeout' else 0.0,
    }
    return sum(terms.values()),terms


def aggregate(rows):
    if not rows:raise ValueError('cannot score an empty panel')
    returns=sorted(float(r['return']) for r in rows)
    if not all(math.isfinite(v) for v in returns):raise ValueError('nonfinite rollout return')
    n=len(rows);finished=[r for r in rows if r['reason']=='finish']
    rate=len(finished)/n
    mean=sum(returns)/n
    tail=returns[:max(1,math.ceil(n*.25))]
    cvar=sum(tail)/len(tail)
    contact=sum(r['wrong_contact_fraction'] for r in rows)/n
    violations={'finish_shortfall':max(0,.8-rate),'contact_excess':max(0,contact-.005)}
    penalty=150*violations['finish_shortfall']+500*violations['contact_excess']
    return {'score':mean+.25*cvar-penalty,'mean_return':mean,'cvar25_return':cvar,
        'constraint_penalty':penalty,'constraints':violations,
        'feasible':rate>=.8 and contact<=.005,'finishers':len(finished),'cases':n,'finish_rate':rate,
        'mean_distance_m':sum(max(0,min(100,r['distance_m'])) for r in rows)/n,
        'wrong_contact_fraction':contact,
        'mean_finish_s':sum(r['time_s'] for r in finished)/len(finished) if finished else None,
        'fall_events':sum(r.get('fall_events',0) for r in rows),
        'recoveries':sum(r.get('recoveries',0) for r in rows),
        'recovery_failures':sum(r['reason']=='fall' for r in rows)}


def objective_spec():
    return {'version':2,'reward':asdict(DEFAULT_REWARD),'cvar_fraction':.25,'cvar_weight':.25,
            'required_finish_rate':.8,'max_wrong_contact_fraction':.005,
            'strict_single_support_requires_zero_contact_frames':True}
