"""Fit height/pitch/omega -> hip and knee targets through official KNP STDP."""
import hashlib
import json
from pathlib import Path
from train_resource_lean import learn_weights

AXES={'height_m':[.15,.4,.65,.95,1.3],
      'pitch_rad':[-3.14,-2,-1,-.5,0,.5,1,2,3.14],
      'angular_speed_rad_s':[-5,-2,0,2,5]}
HEADS=('hip_target_rad','knee_target_rad')


def train_motion(teacher,output):
    """Experimental elapsed-time targets; successful get-up is checked separately."""
    output=Path(output)
    target=[.05+.35*(v+2.8)/5.6 for row in teacher['joint_target_rad'] for v in row]
    report=learn_weights(target,[.225]*len(target))
    decoded=[max(-2.8,min(2.8,(w-.05)/.35*5.6-2.8)) for w in report['weights']]
    policy={**teacher,'joint_target_rad':[decoded[i:i+4] for i in range(0,len(decoded),4)]}
    output.write_text(json.dumps(policy,indent=2))
    report.update(learning='official KNP C++ resource STDP',
        inference='native time interpolation with pitch/omega feedback',
        status='experimental, must pass physical validation before deployment',
        teacher_sha256=hashlib.sha256(json.dumps(teacher,sort_keys=True).encode()).hexdigest())
    output.with_suffix('.checkpoint.json').write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='weights'}


def teacher_values(parameters):
    crouch,stand,pitch_gain,omega_gain=parameters
    heads=[[],[]]
    for height in AXES['height_m']:
        bend=stand+(crouch-stand)*max(0,min(1,(1.15-height)/.8))
        for pitch in AXES['pitch_rad']:
            for omega in AXES['angular_speed_rad_s']:
                heads[0].append(max(-2.6,min(2.6,.5*bend+pitch_gain*pitch+omega_gain*omega)))
                heads[1].append(-bend)
    return heads


def train_recovery(parameters,output,checkpoint=None):
    output=Path(output)
    physical=teacher_values(parameters)
    # Fixed reversible output encoding; plastic weights are changed by KNP only.
    encode=lambda v:.05+.35*(v+2.8)/5.6
    target=[encode(v) for head in physical for v in head]
    initial=[.225]*len(target)
    if checkpoint:
        prior=json.loads(Path(checkpoint).read_text())
        if any(prior[k]!=v for k,v in AXES.items()):raise ValueError('recovery checkpoint axes mismatch')
        initial=[encode(v) for head in HEADS for plane in prior[head] for row in plane for v in row]
    teacher={**AXES,'parameters':parameters,'targets_rad':dict(zip(HEADS,physical))}
    teacher_path=output.with_name('recovery_teacher.json')
    teacher_path.write_text(json.dumps(teacher,indent=2))
    report=learn_weights(target,initial)
    weights=report['weights'];decoded=[max(-2.8,min(2.8,(w-.05)/.35*5.6-2.8)) for w in weights]
    h,p,w=len(AXES['height_m']),len(AXES['pitch_rad']),len(AXES['angular_speed_rad_s'])
    policy=dict(AXES)
    for index,head in enumerate(HEADS):
        start=index*h*p*w
        policy[head]=[[decoded[start+(i*p+j)*w:start+(i*p+j+1)*w] for j in range(p)] for i in range(h)]
    output.write_text(json.dumps(policy,separators=(',',':')))
    report.update(teacher=str(teacher_path),teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                  checkpoint=str(checkpoint) if checkpoint else None,learning='official KNP C++ resource STDP',
                  inference='native trilinear motor target interpolation',encoding='weight=.05+.35*(angle+2.8)/5.6')
    output.with_suffix('.checkpoint.json').write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='weights'}
