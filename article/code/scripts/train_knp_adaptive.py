#!/usr/bin/env python3
"""KNP lean + phase policies selected on diverse native Gymnasium rollouts.

Both neural maps are trained/fine-tuned by official KNP resource STDP. Physical
reward selects teacher parameters; it is not yet online reward-modulated STDP.
Start states are scenarios, never parameters optimized to favor a candidate.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from train_knp_resource_policy import ROOT, target_table
sys.path.insert(0,str(ROOT/'RL/KNP'))
from train_resource_lean import train
from adaptive_rabbit_env import AdaptiveRabbitEnv


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def scenarios(seed,count,level=1.0):
    rng=random.Random(seed);result=[]
    for i in range(count):
        options={'randomize':False}
        kind=i%5
        if kind==1:
            options['initial_lean']=max(.02,min(.3,.19+rng.uniform(-.12,.12)*level))
        elif kind in (2,4):
            options['start_state']={'q_offsets_rad':[rng.uniform(-.06,.06)*level*(1 if j==2 else .35) for j in range(5)],
                'angular_velocity_rad_s':[rng.uniform(-.3,.3)*level for _ in range(5)],
                'linear_velocity_mps':[rng.uniform(-.2,.2)*level,0]}
            if kind==4:
                options['initial_lean']=.19+rng.uniform(-.04,.04)*level
        elif kind==3:
            options.update(push_at_s=rng.uniform(2,8),torso_impulse_ns=[rng.uniform(-.25,.25)*level,0])
        result.append({'seed':seed+i,'kind':['nominal','angle','pose','push','combined'][kind],'options':options})
    return result


def rollout(job):
    path,scenario=job
    # Legacy selection keeps its original immediate-fall protocol.
    env=AdaptiveRabbitEnv(path,repeat_steps=24,horizon=45,recovery_grace_s=0)
    try:
        env.reset(seed=scenario['seed'],options=scenario['options'])
        samples=wrong=0;total=0.0
        while True:
            _,reward,terminated,truncated,info=env.step(np.zeros(5,dtype=np.float32))
            total+=reward;samples+=info['physics_frames'];wrong+=info['wrong_contact_frames']
            if terminated or truncated:break
        return {**info,'scenario':scenario,'wrong_contact_fraction':wrong/samples,'return':total}
    finally:env.close()


def aggregate(rows):
    n=len(rows);finished=[r for r in rows if r['reason']=='finish']
    rate=len(finished)/n
    distance=sum(max(0,min(100,r['distance_m'])) for r in rows)/n
    contact=sum(r['wrong_contact_fraction'] for r in rows)/n
    time=sum(r['time_s'] for r in finished)/len(finished) if finished else 45
    return {'score':100*rate+distance*.1-5*contact-time*.05,'finishers':len(finished),
            'cases':n,'finish_rate':rate,'mean_distance_m':distance,'wrong_contact_fraction':contact,
            'mean_finish_s':time if finished else None}


def train_candidate(index,parameters,axes,base,directory,elite):
    directory=directory/f'candidate-{index:03d}';directory.mkdir(parents=True,exist_ok=True)
    teacher=target_table(axes,parameters[:5]+[0])
    teacher_path=directory/'lean_teacher.json';teacher_path.write_text(json.dumps(teacher))
    lean_path=directory/'lean.json'
    lean_report=train(teacher_path,lean_path,scale=1,bias=0,
                      checkpoint=Path(elite['lean_policy']) if elite else None)
    phase=copy.deepcopy(teacher)
    for i,v in enumerate(axes['speed_mps']):
        for j,pitch in enumerate(axes['pitch_rad']):
            for k,w in enumerate(axes['angular_speed_rad_s']):
                blend=max(0,min(1,parameters[5]+parameters[6]*(v-1.3)+parameters[7]*(pitch+.35)+parameters[8]*w))
                phase['lean_rad'][i][j][k]=.05+.35*blend
    phase_teacher=directory/'phase_teacher.json';phase_teacher.write_text(json.dumps(phase))
    phase_path=directory/'phase.json'
    phase_report=train(phase_teacher,phase_path,scale=1,bias=0,
                       checkpoint=Path(elite['phase_policy']) if elite else None)
    command=copy.deepcopy(base)
    command.update(initial_lean=.19,startup_blend_s=0.0,start_state=None,
                   knp_lean_sensor_table=json.loads(lean_path.read_text()),
                   knp_phase_sensor_table=json.loads(phase_path.read_text()))
    command['stance_kp']*=parameters[9];command['stance_kd']*=parameters[10]
    config=directory/'command.json';config.write_text(json.dumps(command))
    return {'index':index,'parameters':parameters,'command':str(config),'lean_policy':str(lean_path),
            'phase_policy':str(phase_path),'knp_lean':lean_report,'knp_phase':phase_report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--viewer-pid',type=int,required=True)
    parser.add_argument('--population',type=int,default=12)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--train-cases',type=int,default=10)
    parser.add_argument('--selection-cases',type=int,default=20)
    parser.add_argument('--generations',type=int,default=0)
    args=parser.parse_args()
    if args.population<3 or not 1<=args.workers<=32 or min(args.train_cases,args.selection_cases)<4 or args.generations<0:
        parser.error('population>=3, cases>=4, workers 1..32, generations>=0 required')
    manifest=json.loads(args.manifest.read_text());root=args.manifest.parent
    base=json.loads((root/'base_payload.json').read_text())
    axes=json.loads((ROOT/'RL/KNP/experiments/knp_running_lean_sensors.json').read_text())
    directory=root/'adaptive_training';directory.mkdir(exist_ok=True)
    elite=json.loads((directory/'best.json').read_text()) if (directory/'best.json').exists() else None
    default=[.20660809579629938,.023265770923763723,.13258223786373838,.09304505701504834,.007808660109614918,0,0,0,0,1,1]
    bounds=[(.12,.26),(0,.07),(0,.3),(-.08,.2),(-.01,.03),(0,1),(-1,1),(-2,2),(-.25,.25),(.4,2),(.4,2)]
    sigmas=[.007,.006,.02,.02,.002,.3,.25,.4,.05,.15,.15]
    generation=max([manifest.get('generation',0)]+[int(p.name.split('-')[-1]) for p in directory.glob('generation-*')])
    last_generation=generation+args.generations if args.generations else None
    selection=scenarios(20261110,args.selection_cases,1.0)
    panel_hash=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if elite and elite.get('selection_panel_sha256')!=panel_hash:
            rows=list(pool.map(rollout,[(elite['command'],s) for s in selection]))
            elite.update(selection=aggregate(rows),selection_trials=rows,selection_panel_sha256=panel_hash)
            print(json.dumps({'resumed_selection':elite['selection'],'panel_sha256':panel_hash}),flush=True)
        while alive(args.viewer_pid) and (last_generation is None or generation<last_generation):
            generation+=1;rng=random.Random(61773+generation)
            generation_dir=directory/f'generation-{generation:05d}';generation_dir.mkdir()
            center=elite['parameters'] if elite else default
            level=.35 if not elite else min(1.0,.35+elite['selection']['finish_rate']*.65)
            training=scenarios(90000+generation*37,args.train_cases,level)
            (generation_dir/'scenarios.json').write_text(json.dumps({'training':training,'selection':selection,'level':level},indent=2))
            parameters=[center.copy(),center.copy(),default.copy()]
            parameters[1][5:9]=[1.0 if center[5]<.5 else 0.0,0,0,0]
            parameters[2][5:9]=[1,0,0,0]
            for _ in range(args.population-3):
                parameters.append([max(lo,min(hi,x+rng.gauss(0,s))) for x,s,(lo,hi) in zip(center,sigmas,bounds)])
            candidates=[]
            for i,p in enumerate(parameters):
                if not alive(args.viewer_pid):return
                try:
                    candidate=train_candidate(i,p,axes,base,generation_dir,elite)
                except RuntimeError as error:
                    failure={'generation':generation,'candidate':i,'error':str(error)}
                    (generation_dir/f'candidate-{i:03d}'/'failure.json').write_text(json.dumps(failure,indent=2))
                    print(json.dumps(failure),flush=True)
                    continue
                candidate['generation']=generation
                rows=list(pool.map(rollout,[(candidate['command'],s) for s in training]))
                candidate['training']=aggregate(rows);candidate['train_trials']=rows;candidates.append(candidate)
                print(json.dumps({'generation':generation,'candidate':i,**candidate['training']}),flush=True)
                (generation_dir/'results.json').write_text(json.dumps(candidates,indent=2))
            if not candidates:raise RuntimeError('all KNP candidates failed; see generation failure.json files')
            finalists=sorted(candidates,key=lambda c:c['training']['score'],reverse=True)[:3]
            for candidate in finalists:
                rows=list(pool.map(rollout,[(candidate['command'],s) for s in selection]))
                candidate['selection']=aggregate(rows);candidate['selection_trials']=rows
                candidate['selection_panel_sha256']=panel_hash
            winner=max(finalists,key=lambda c:c['selection']['score'])
            if elite is None or winner['selection']['score']>elite['selection']['score']:
                elite=winner
            (directory/'best.json').write_text(json.dumps(elite,indent=2))
            (generation_dir/'results.json').write_text(json.dumps(candidates,indent=2))
            (generation_dir/'best.json').write_text(json.dumps(elite,indent=2))
            command=json.loads(Path(elite['command']).read_text())
            for i,lane in enumerate(manifest['lanes']):
                if lane['name']=='baseline-hybrid':continue
                lane['name']=f'adaptive-g{generation:05d}'
                lane['command']=command
                lane['randomization']={'seed':710000+generation,'lean_range':.04*level,
                    'pose_range':.06*level,'speed_range':.2*level,'omega_range':.3*level}
            manifest['generation']=generation
            encoded=json.dumps(manifest,separators=(',',':'))
            with gzip.open(generation_dir/'published_manifest.json.gz','wt') as out:out.write(encoded)
            temp=args.manifest.with_suffix('.tmp');temp.write_text(encoded);temp.replace(args.manifest)
            status={'generation':generation,'learning':'KNP lean+phase resource STDP, randomized physics selection',
                    'best_generation':elite['generation'],
                    'selection':elite['selection'],'training_level':level,'neurons_per_map':315,
                    'best_command':elite['command']}
            (root/'training_status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)


if __name__=='__main__':main()
