#!/usr/bin/env python3
"""KNP STDP policies: strong physical costs, covariance search, paired validation.

This is reward-based teacher search and STDP distillation, not PPO or CPO.
Recovery is a separately observed motor skill with a finite opportunity to stand.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from train_knp_adaptive import ROOT, alive, scenarios, train_candidate
from knp_risk_search import CovarianceSearch
from knp_policy_promotion import validation_panel,panel_hash,promotion_gate,publish_champion
sys.path.insert(0,str(ROOT/'RL/KNP'))
from adaptive_rabbit_env import AdaptiveRabbitEnv
from locomotion_objective import aggregate, objective_spec
from train_recovery_policy import train_recovery


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def rollout(job):
    path,scenario,*duration=job
    env=AdaptiveRabbitEnv(path,repeat_steps=24,horizon=duration[0] if duration else 45,recovery_grace_s=8)
    try:
        env.reset(seed=scenario['seed'],options=scenario['options'])
        frames=wrong=recovery_frames=0;total=0.;terms={}
        while True:
            _,reward,terminated,truncated,info=env.step(np.zeros(5,dtype=np.float32))
            total+=reward;frames+=info['physics_frames'];wrong+=info['wrong_contact_frames']
            recovery_frames+=info.get('recovery_frames',0)
            for key,value in info['reward_terms'].items():terms[key]=terms.get(key,0)+value
            if terminated or truncated:break
        return {**info,'scenario':scenario,'wrong_contact_fraction':wrong/max(1,frames-recovery_frames),
                'wrong_contact_total_frames':wrong,'physics_total_frames':frames,
                'recovery_time_s':recovery_frames/240,'return':total,'reward_totals':terms}
    finally:env.close()


def enrich_candidate(candidate,parameters,directory,prior=None):
    path=directory/'recovery.json'
    if prior and prior.get('recovery_motion'):
        # Keep the validated get-up skill exact while optimizing locomotion.
        # Its preparation map is part of the skill, including the delay before
        # the independent left/right motion starts.
        parameters=list(parameters);parameters[11:]=prior['parameters'][11:]
        atomic_json(path,json.loads(Path(prior['recovery_policy']).read_text()))
        report={**prior['knp_recovery'],'inherited_validated_skill':True}
    else:
        report=train_recovery(parameters[11:],path,prior.get('recovery_policy') if prior else None)
    command=json.loads(Path(candidate['command']).read_text())
    command.update(recovery_grace_s=8,knp_recovery_policy=json.loads(path.read_text()))
    if prior and prior.get('recovery_motion'):
        command.update(knp_recovery_motion=json.loads(Path(prior['recovery_motion']).read_text()),
                       recovery_pd=prior['recovery_pd'])
        candidate.update(recovery_motion=prior['recovery_motion'],recovery_pd=prior['recovery_pd'],
                         recovery_validation=prior['recovery_validation'])
    command_path=directory/'command.json';atomic_json(command_path,command)
    candidate.update(parameters=parameters,command=str(command_path),recovery_policy=str(path),knp_recovery=report)
    return candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--viewer-pid',type=int,required=True)
    parser.add_argument('--seed-best',type=Path,default=ROOT/'RL/KNP/experiments/swarm/adaptive-live-20261110/adaptive_training/best.json')
    parser.add_argument('--population',type=int,default=16)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--generations',type=int,default=0)
    args=parser.parse_args()
    if not 4<=args.population<=100 or not 1<=args.workers<=32 or args.generations<0:
        parser.error('population 4..100, workers 1..32, generations nonnegative')
    root=args.manifest.resolve().parent
    def progress(stage,**details):
        atomic_json(root/'training_progress.json',{'stage':stage,'updated_unix_s':time.time(),'pid':os.getpid(),**details})
    progress('initializing')
    manifest=json.loads(args.manifest.read_text());base=json.loads((root/'base_payload.json').read_text())
    directory=root/'risk_training';directory.mkdir(exist_ok=True)
    axes=json.loads((ROOT/'RL/KNP/experiments/knp_running_lean_sensors.json').read_text())
    atomic_json(directory/'objective.json',objective_spec())
    if (directory/'best.json').exists():elite=json.loads((directory/'best.json').read_text())
    else:
        prior=json.loads(args.seed_best.read_text());seed_dir=directory/'seed';seed_dir.mkdir(exist_ok=True)
        elite=enrich_candidate(copy.deepcopy(prior),prior['parameters'][:11]+[1.7,.25,1.3,.22],seed_dir)
        elite.update(generation=0,index=0,source_checkpoint=str(args.seed_best.resolve()))
        atomic_json(directory/'best.json',elite)
    previous=json.loads((directory/'search.json').read_text()) if (directory/'search.json').exists() else {}
    search=CovarianceSearch(elite['parameters'],previous.get('distribution'))
    replay=previous.get('replay',[])
    fixed_panel=validation_panel(961111)
    fixed_cache={}
    generation=max([manifest.get('generation',0)]+[int(p.name.split('-')[-1]) for p in directory.glob('generation-*')])
    expected_name=f"risk-g{elite['generation']:05d}"
    if any(lane['name'] not in ('baseline-hybrid',expected_name) for lane in manifest['lanes']):
        # Repair old labels that used the attempted generation instead of the
        # champion's generation. This changes no weights or start conditions.
        generation=max(generation,manifest.get('generation',0)+1)
        publish_champion(manifest,elite,json.loads(Path(elite['command']).read_text()),generation)
        atomic_json(args.manifest,manifest)
    stop_generation=generation+args.generations if args.generations else None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def evaluate(candidate,panel,key):
            rows=list(pool.map(rollout,[(candidate['command'],s,manifest['episode_seconds']) for s in panel]))
            candidate[key]=aggregate(rows);candidate[key+'_trials']=rows
        def qualify(candidate,incumbent,fresh_panel):
            gates={}
            for name,panel in [('fixed',fixed_panel),('fresh',fresh_panel)]:
                results=[]
                for policy in (incumbent,candidate):
                    digest=hashlib.sha256(Path(policy['command']).read_bytes()).hexdigest()
                    if name=='fixed' and digest in fixed_cache:rows=fixed_cache[digest]
                    else:
                        rows=list(pool.map(rollout,[(policy['command'],s,manifest['episode_seconds']) for s in panel]))
                        if name=='fixed':fixed_cache[digest]=rows
                    results.append(rows)
                gates[name]=promotion_gate(*results)
                atomic_json(run/f'guard-{candidate["index"]:03d}-{name}.json',{
                    'panel_sha256':panel_hash(panel),'gate':gates[name],
                    'incumbent_command':incumbent['command'],'candidate_command':candidate['command'],
                    'incumbent_trials':results[0],'candidate_trials':results[1]})
                if not gates[name]['accepted']:break
            candidate['promotion_gates']=gates
            return len(gates)==2 and all(g['accepted'] for g in gates.values())
        while alive(args.viewer_pid) and (stop_generation is None or generation<stop_generation):
            generation+=1;run=directory/f'generation-{generation:05d}';run.mkdir()
            # Both policies are tested on exactly the same fresh validation panel.
            level=min(1.,.5+.5*elite.get('selection',{}).get('finish_rate',.5))
            training=[s for i,s in enumerate(scenarios(310000+generation*71,50,level))
                      if i==0 or s['kind']!='nominal']+replay[:8]
            selection=scenarios(811711,5,1.)+scenarios(812000+generation*97,15,1.)
            fresh_panel=validation_panel(981111+generation*131)
            atomic_json(run/'scenarios.json',{'training':training,'selection':selection,'level':level})
            params=search.sample(args.population,51000+generation,elite['parameters'])
            candidates=[]
            for i,parameters in enumerate(params):
                if not alive(args.viewer_pid):return
                progress('KNP STDP',generation=generation,candidate=i+1,population=len(params))
                if i==0:
                    # Keep exact incumbent weights: no numerical re-fitting before comparison.
                    candidate=copy.deepcopy(elite);candidate['index']=0
                else:
                    try:
                        candidate=train_candidate(i,parameters[:11],axes,base,run,elite)
                        candidate=enrich_candidate(candidate,parameters,run/f'candidate-{i:03d}',elite)
                        candidate['generation']=generation
                    except RuntimeError as error:
                        atomic_json(run/f'failure-{i:03d}.json',{'error':str(error)})
                        print(json.dumps({'generation':generation,'candidate':i,'error':str(error)}),flush=True)
                        continue
                progress('physics trials',generation=generation,candidate=i+1,population=len(params))
                evaluate(candidate,training,'training');candidates.append(candidate)
                progress('candidate scored',generation=generation,candidate=i+1,population=len(params),**candidate['training'])
                atomic_json(run/'results.json',candidates)
                print(json.dumps({'generation':generation,'candidate':i,**candidate['training']}),flush=True)
            if not candidates:raise RuntimeError('no completed candidates')
            incumbent=candidates[0]
            finalists=sorted(candidates[1:],key=lambda c:(c['training']['finishers'],
                -c['training']['fall_events'],c['training']['score']),reverse=True)[:3]
            for index,candidate in enumerate([incumbent]+finalists):
                progress('paired validation',generation=generation,candidate=index+1,population=1+len(finalists))
                evaluate(candidate,selection,'selection')
            elite=incumbent;improved=False
            for candidate in sorted(finalists,key=lambda c:c['selection']['score'],reverse=True):
                a,b=candidate['selection'],incumbent['selection']
                if a['score']<=b['score'] or a['finishers']<b['finishers'] or a['fall_events']>b['fall_events']:
                    continue
                progress('promotion guard: 2 x 101',generation=generation,candidate=candidate['index']+1,
                         population=len(params))
                if qualify(candidate,incumbent,fresh_panel):
                    elite=candidate;improved=True;break
            # Bound the cache to the current champion; discarded candidates must
            # not accumulate indefinitely in a continuously running trainer.
            elite_digest=hashlib.sha256(Path(elite['command']).read_bytes()).hexdigest()
            fixed_cache={k:v for k,v in fixed_cache.items() if k==elite_digest}
            search.update(candidates,improved)
            # Only training failures enter replay; validation/test panels never do.
            difficult=sorted(elite['training_trials'],key=lambda r:r['return'])
            unique={json.dumps(s,sort_keys=True):s for s in [r['scenario'] for r in difficult]+replay}
            replay=list(unique.values())[:24]
            atomic_json(directory/'search.json',{'distribution':search.state(),'replay':replay,'generation':generation})
            atomic_json(directory/'best.json',elite);atomic_json(run/'results.json',candidates);atomic_json(run/'best.json',elite)
            if improved:
                command=json.loads(Path(elite['command']).read_text())
                publish_champion(manifest,elite,command,generation)
                atomic_json(args.manifest,manifest)
            with gzip.open(run/'published_manifest.json.gz','wt') as out:json.dump(manifest,out)
            status={'generation':generation,'best_generation':elite['generation'],'best_command':elite['command'],
                'learning':'KNP lean/phase/recovery STDP + covariance CEM + return/CVaR/constraints',
                'selection':elite['selection'],'improved':improved,'training_level':level,
                'promotion_rule':'paired fixed + fresh 101 cases; no lost finishes, no extra falls by kind; contact and lower-tail guards',
                'promotion_gates':{str(c['index']):c.get('promotion_gates',{}) for c in finalists},
                'replay_cases':len(replay),'neurons':1080+(28 if elite.get('recovery_motion') else 0),
                'episode_seconds':manifest['episode_seconds'],'requested_speed':manifest['speed'],
                'selection_panel_sha256':hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()}
            atomic_json(root/'training_status.json',status);print(json.dumps(status),flush=True)
            progress('published' if improved else 'champion retained',generation=generation,**elite['selection'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        if len(sys.argv)>1 and Path(sys.argv[1]).is_file():
            atomic_json(Path(sys.argv[1]).resolve().parent/'training_progress.json',
                        {'stage':'FAILED','detail':str(error)[:120],'updated_unix_s':time.time(),'pid':os.getpid()})
        raise
