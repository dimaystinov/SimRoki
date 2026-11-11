"""Conservative paired deployment gate. Exploration never directly replaces a champion."""
import hashlib
import json
import random
import numpy as np

from train_knp_adaptive import scenarios
from locomotion_objective import aggregate

GATE_VERSION=1


def validation_panel(seed,cases_per_kind=25):
    # Nominal is deterministic: repeating it would falsely increase sample size.
    panel=scenarios(seed,cases_per_kind*5,1.)
    rng=random.Random(seed ^ 721)
    for s in panel:
        if s['kind']=='angle':s['options']['initial_lean']=rng.uniform(.05,.29)
    return [s for i,s in enumerate(panel) if i==0 or s['kind']!='nominal']


def panel_hash(panel):
    return hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()


def promotion_gate(incumbent,candidate):
    reasons=[]
    report={'version':GATE_VERSION,'accepted':False,'reasons':reasons,'cases':len(candidate)}
    if len(candidate)<101 or len(incumbent)!=len(candidate):
        reasons.append('insufficient_paired_cases');return report
    if any(a['scenario']!=b['scenario'] for a,b in zip(incumbent,candidate)):
        reasons.append('scenario_mismatch');return report
    keys=[json.dumps(r['scenario']['options'],sort_keys=True) for r in candidate]
    if len(set(keys))!=len(keys):
        reasons.append('duplicate_conditions');return report
    kinds=sorted({r['scenario']['kind'] for r in incumbent})
    if any(sum(r['scenario']['kind']==k for r in candidate)<25 for k in ('angle','pose','push','combined')):
        reasons.append('insufficient_scenario_coverage');return report
    old,new=aggregate(incumbent),aggregate(candidate)
    report.update(incumbent=old,candidate=new,by_kind={})
    for kind in kinds:
        pairs=[(a,b) for a,b in zip(incumbent,candidate) if a['scenario']['kind']==kind]
        losses=sum(a['reason']=='finish' and b['reason']!='finish' for a,b in pairs)
        gains=sum(a['reason']!='finish' and b['reason']=='finish' for a,b in pairs)
        a,b=aggregate([a for a,_ in pairs]),aggregate([b for _,b in pairs])
        report['by_kind'][kind]={'lost_finishes':losses,'gained_finishes':gains,
            'incumbent':a,'candidate':b}
        if losses:reasons.append('lost_finish:'+kind)
        if b['fall_events']>a['fall_events']:reasons.append('extra_falls:'+kind)
        if b['wrong_contact_fraction']>a['wrong_contact_fraction']+1e-6:
            reasons.append('more_wrong_contact:'+kind)
    if new['cvar25_return']<old['cvar25_return']-1e-6:reasons.append('worse_lower_tail')
    if new['score']<=old['score']+1e-6:reasons.append('no_score_improvement')
    # Common scenarios permit paired resampling. This is a finite-panel guard,
    # not a guarantee for unseen poses or a sequential statistical proof.
    delta=np.array([b['return']-a['return'] for a,b in zip(incumbent,candidate)])
    rng=np.random.default_rng(80261111)
    means=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
    report['paired_return_gain']=float(delta.mean())
    report['paired_return_gain_lower95']=float(np.quantile(means,.05))
    if report['paired_return_gain_lower95']<=1e-6:reasons.append('uncertain_return_gain')
    report['accepted']=not reasons
    return report


def publish_champion(manifest,elite,command,generation):
    for lane in manifest['lanes']:
        if lane['name']=='baseline-hybrid':continue
        lane.update(name=f"risk-g{elite['generation']:05d}",command=command)
        if lane.get('randomization') is None:
            lane['randomization']={'seed':910000,'lean_range':.04,'pose_range':.06,
                                   'speed_range':.2,'omega_range':.3}
        # Keep the exact seed and difficulty. Curriculum changes belong to
        # training, not to the visual A/B comparison at every publication.
    manifest['generation']=generation


def recovery_module_gate(incumbent_command,candidate_command,incumbent,candidate):
    """Recovery-only change: compare first falls and successful post-fall runs.

    Total falls are censored by the old policy ending immediately after failed
    recovery. Report repeat falls explicitly; never call this a first-fall gain.
    The stricter locomotion replacement gate remains unchanged.
    """
    report=promotion_gate(incumbent,candidate)
    reasons=[]
    changed={k for k in set(incumbent_command)|set(candidate_command)
             if incumbent_command.get(k)!=candidate_command.get(k)}
    if not changed or not changed<={'recovery_pd','knp_recovery_motion'}:
        reasons.append('not_recovery_only')
    if not candidate_command.get('knp_recovery_motion'):reasons.append('missing_recovery_motion')
    if 'by_kind' not in report:
        return {**report,'accepted':False,'recovery_only':True}
    for kind in report['by_kind']:
        pairs=[(a,b) for a,b in zip(incumbent,candidate) if a['scenario']['kind']==kind]
        if sum(b.get('fall_events',0)>0 for _,b in pairs)>sum(a.get('fall_events',0)>0 for a,_ in pairs):
            reasons.append('more_first_falls:'+kind)
        if sum(b['reason']=='fall' for _,b in pairs)>sum(a['reason']=='fall' for a,_ in pairs):
            reasons.append('more_unrecovered_falls:'+kind)
        for a,b in pairs:
            if a['reason']=='finish' and (b['reason']!='finish' or abs(a['return']-b['return'])>1e-5):
                reasons.append('changed_previous_success:'+kind);break
    recovered_finishes=sum(r.get('recoveries',0)>0 and r['reason']=='finish' for r in candidate)
    old_recovered=sum(r.get('recoveries',0)>0 and r['reason']=='finish' for r in incumbent)
    if recovered_finishes<=old_recovered:reasons.append('no_additional_completed_recovery')
    if report['candidate']['score']<=report['incumbent']['score']:reasons.append('no_score_improvement')
    if report['paired_return_gain_lower95']<=0:reasons.append('uncertain_return_gain')
    if report['candidate']['recovery_failures']>=report['incumbent']['recovery_failures']:
        reasons.append('no_recovery_failure_reduction')
    return {**report,'accepted':not reasons,'reasons':reasons,'recovery_only':True,
            'changed_fields':sorted(changed),'recovered_finishes':recovered_finishes,
            'repeat_falls':sum(max(0,r.get('fall_events',0)-1) for r in candidate),
            'first_fall_reduction_claimed':False}
