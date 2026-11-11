#!/usr/bin/env python3
"""Discover a recovery teacher in native physics, then distill through KNP STDP.

The search uses no external neural learner. A candidate is not deployed merely
because it moves or reaches a high torso position: recovery and resumed running
are recorded separately. This experiment does not replace the race champion.
"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'RL/KNP'))
from adaptive_rabbit_env import AdaptiveRabbitEnv
from train_recovery_policy import train_motion

PHASES=[0.,.4,1.,1.8,2.8,4.2,6.]


def command_from_parameters(base,p):
    c=dict(base)
    c['recovery_pd']=[12.,.15]
    c['knp_recovery_motion']={'start_after_s':float(p[-3]),'phase_s':PHASES,
        'joint_target_rad':np.asarray(p[:-3]).reshape(-1,4).tolist(),
        'pitch_gain':float(p[-2]),'omega_gain':float(p[-1])}
    return c


def evaluate(job):
    base,scenario,p=job
    command=command_from_parameters(base,p)
    return {**evaluate_command(command,scenario),'parameters':p}


def evaluate_command(command,scenario):
    e=AdaptiveRabbitEnv(command,repeat_steps=12,horizon=60,recovery_grace_s=8)
    e.reset(seed=scenario['seed'],options=scenario['options'])
    quality=[];best_window=0.;max_stable=0.;recovered_x=None;resume_distance=0.;resume_time=0.
    peak_height=0.;peak_usable=0.;contacts=0;started=False
    try:
        while True:
            _,_,term,trunc,info=e.step(np.zeros(5))
            s=e.snapshot['state'];b=s['base'];r=e.snapshot['recovery']
            if r['active']:
                started=True
                # Discovery reward preserves gradients toward a supported upright
                # posture even before the first successful 0.3-s stable hold.
                support=any(s['contacts'].values())
                height=min(1.3,max(0,b['y']))/1.3
                upright=max(0,np.cos(b['angle']))**2
                usable=height*upright*np.exp(-.035*b['omega']**2)*(.7+.3*support)
                quality.append(float(usable))
                if len(quality)>=4:best_window=max(best_window,float(np.mean(quality[-4:])))
                max_stable=max(max_stable,r['stable_s']);peak_height=max(peak_height,b['y'])
                peak_usable=max(peak_usable,float(usable));contacts+=int(support)
            elif started:
                if recovered_x is None:recovered_x=b['x']
                resume_distance=max(resume_distance,b['x']-recovered_x);resume_time+=.05
            if term or trunc:break
        # Repeated falling/standing must never beat one recovery followed by a
        # completed run. Count the skill once and penalize every subsequent fall.
        score=(10*best_window+peak_usable+20*max_stable+100*int(info['recoveries']>0)
               +min(30,resume_distance)+min(10,resume_time)
               +200*int(info['reason']=='finish')-50*max(0,info['fall_events']-1))
        return {'score':score,'recoveries':info['recoveries'],'fall_events':info['fall_events'],
            'resume_distance_m':resume_distance,'resume_time_s':resume_time,'reason':info['reason'],
            'max_stable_s':max_stable,'peak_height_m':peak_height,'peak_usable':peak_usable,
            'best_window':best_window,'contacts':contacts,'time_s':info['time_s'],
            'distance_m':info['distance_m']}
    finally:e.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--generations',type=int,default=40)
    parser.add_argument('--population',type=int,default=80)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--seed',type=int,default=11012026)
    parser.add_argument('--seed-command',type=Path)
    parser.add_argument('--scenarios',type=Path,help='fixed discovery bank; independent validation stays separate')
    args=parser.parse_args();directory=args.directory.resolve()
    base=json.loads((directory/'base_command.json').read_text())
    scenario=json.loads((directory/'scenario.json').read_text())
    scenarios=json.loads(args.scenarios.read_text()) if args.scenarios else [scenario]
    low=np.array([-2.7,-2.7,-2.7,-2.7]*len(PHASES)+[0.,-2.5,-.5])
    high=np.array([2.7,.1,2.7,.1]*len(PHASES)+[2.,2.5,.5])
    initial=np.array([[1.1,-2.2,1.1,-2.2],[1.1,-2.2,1.1,-2.2],
        [.7,-1.4,.7,-1.4],[.25,-.5,.25,-.5],[.1,-.2,.1,-.2],
        [.1,-.2,.1,-.2],[.1,-.2,.1,-.2]]).flatten().tolist()+[0.,-1.,.15]
    if args.seed_command:
        motion=json.loads(args.seed_command.read_text())['knp_recovery_motion']
        if motion['phase_s']!=PHASES:raise ValueError('seed motion phase axis mismatch')
        initial=[v for row in motion['joint_target_rad'] for v in row]+[motion['start_after_s'],motion['pitch_gain'],motion['omega_gain']]
    mean=np.array(initial);std=np.array([.7,.55,.7,.55]*len(PHASES)+[.5,.65,.15])
    if args.seed_command:std*=.25
    rng=np.random.default_rng(args.seed);best=None;history=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for generation in range(args.generations):
            parameters=[mean.tolist(),best['parameters'] if best else initial]
            for i in range(args.population-2):
                scale=2. if i>=args.population-12 else 1.
                center=initial if i>=args.population-6 else mean
                parameters.append(np.clip(center+rng.normal(0,std*scale),low,high).tolist())
            trials=list(pool.map(evaluate,[(base,s,p) for p in parameters for s in scenarios]))
            rows=[]
            for i,p in enumerate(parameters):
                group=trials[i*len(scenarios):(i+1)*len(scenarios)]
                rows.append({'score':float(np.mean([r['score'] for r in group])),
                    'finishers':sum(r['reason']=='finish' for r in group),'cases':len(group),
                    'recoveries':sum(r['recoveries'] for r in group),
                    'fall_events':sum(r['fall_events'] for r in group),
                    'parameters':p,'trials':group})
            rows.sort(key=lambda r:r['score'],reverse=True)
            changed=best is None or rows[0]['score']>best['score']
            if changed:best=rows[0]
            top=np.array([r['parameters'] for r in rows[:max(8,args.population//8)]])
            mean=.2*mean+.8*top.mean(axis=0);std=np.maximum(.05,.4*std+.6*top.std(axis=0))
            status={'generation':generation,'best':best,'teacher_search':True,'deployed':False}
            history.append(rows)
            (directory/'search.json').write_text(json.dumps({'status':status,'history':history}))
            (directory/'progress.json').write_text(json.dumps(status,indent=2))
            (directory/'teacher_command.json').write_text(json.dumps(command_from_parameters(base,best['parameters'])))
            print(json.dumps({k:v for k,v in best.items() if k not in ('parameters','trials')}|{'generation':generation}),flush=True)
            if changed:
                command=command_from_parameters(base,best['parameters'])
                report=train_motion(command['knp_recovery_motion'],directory/'knp_motion.json')
                command['knp_recovery_motion']=json.loads((directory/'knp_motion.json').read_text())
                learned=list(pool.map(evaluate,[(base,s,
                    [v for row in command['knp_recovery_motion']['joint_target_rad'] for v in row]
                    +best['parameters'][-3:]) for s in scenarios]))
                (directory/'knp_command.json').write_text(json.dumps(command))
                (directory/'knp_validation.json').write_text(json.dumps({'learning':report,'trials':learned},indent=2))
    # All exported neural motor targets are plastic weights learned by official KNP.
    command=command_from_parameters(base,best['parameters'])
    report=train_motion(command['knp_recovery_motion'],directory/'knp_motion.json')
    command['knp_recovery_motion']=json.loads((directory/'knp_motion.json').read_text())
    (directory/'knp_command.json').write_text(json.dumps(command))
    (directory/'learning.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
