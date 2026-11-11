"""Reproduce manuscript statistics and vector figures from frozen project artifacts.

Run: python3 analyze.py
Requires numpy and matplotlib. Does not run or train the robot.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA, FIG = ROOT / 'data', ROOT / 'figures'
FIG.mkdir(exist_ok=True)
audit = json.loads((DATA / 'independent_audit.json').read_text())
old, new = audit[:2]
assert len(old['trials']) == len(new['trials']) == 101
for a, b in zip(old['trials'], new['trials']):
    assert a['scenario'] == b['scenario']
delta = np.array([b['return']-a['return'] for a,b in zip(old['trials'],new['trials'])])
gained = sum(a['reason']!='finish' and b['reason']=='finish' for a,b in zip(old['trials'],new['trials']))
lost = sum(a['reason']=='finish' and b['reason']!='finish' for a,b in zip(old['trials'],new['trials']))
assert (gained, lost) == (22, 0)
assert all(a == b for a,b in zip(old['trials'],new['trials']) if a['reason']=='finish')
rng = np.random.default_rng(80261111)
boot = delta[rng.integers(0, len(delta), size=(2000,len(delta)))].mean(axis=1)
results = {'n':101, 'gained_finishes':gained, 'lost_finishes':lost,
           'finish_gain_pp':100*gained/101, 'mean_return_gain':float(delta.mean()),
           'paired_return_gain_lower95_one_sided':float(np.quantile(boot,.05)),
           'bootstrap_resamples':2000, 'bootstrap_seed':80261111,
           'interpretation':'Exploratory paired validation panel used for module acceptance; not an external test.'}
names = ['nominal','angle','pose','push','combined']
labels = ['Номинал','Наклон','Поза','Толчок','Сочетание']
rows = []
for name in names:
    row={'kind':name}
    for tag,pol in [('base',old),('r009',new)]:
        ts=[t for t in pol['trials'] if t['scenario']['kind']==name]
        row['n']=len(ts)
        row[tag+'_finish']=sum(t['reason']=='finish' for t in ts)
        row[tag+'_first_fall']=sum(t['fall_events']>0 for t in ts)
        row[tag+'_terminal_failure']=sum(t['reason']=='fall' for t in ts)
    rows.append(row)
results['by_kind']=rows
for tag,pol in [('base',old),('r009',new)]:
    results[tag] = dict(pol['overall'])
    results[tag]['outcomes'] = dict(collections.Counter(t['reason'] for t in pol['trials']))
    results[tag]['first_fall_episodes']=sum(t['fall_events']>0 for t in pol['trials'])
    results[tag]['finish_after_recovery']=sum(t['reason']=='finish' and t['recoveries']>0 for t in pol['trials'])
    results[tag]['repeat_falls']=sum(max(0,t['fall_events']-1) for t in pol['trials'])
(ROOT/'statistics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'paired_outcomes.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['seed','kind','base_reason','r009_reason','base_falls','r009_falls','base_recoveries','r009_recoveries','base_return','r009_return','base_wrong_fraction','r009_wrong_fraction'])
    for a,b in zip(old['trials'],new['trials']):
        writer.writerow([a['scenario']['seed'],a['scenario']['kind'],a['reason'],b['reason'],a['fall_events'],b['fall_events'],a['recoveries'],b['recoveries'],a['return'],b['return'],a['wrong_contact_fraction'],b['wrong_contact_fraction']])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.bbox':'tight','axes.edgecolor':'#69727b',
    'xtick.color':'#38424b','ytick.color':'#38424b'})
blue, grey, red = '#204e70','#98a2aa','#b14e46'

fig,ax=plt.subplots(figsize=(6.5,2.6))
x=np.arange(4);width=.34
v0=[69,29,29,0];v1=[91,29,6,22]
for dx,vals,col,lab in [(-width/2,v0,grey,'Базовое восстановление'),(width/2,v1,blue,'KNP r009')]:
    bars=ax.bar(x+dx,vals,width,color=col,label=lab)
    ax.bar_label(bars,padding=3,fontsize=9)
ax.set_xticks(x,['Финиш\n100 м','Хотя бы одно\nпадение','Неудачное\nвосстановление','Финиш после\nвосстановления'])
ax.set_ylim(0,108);ax.set_ylabel('Число эпизодов из 101')
ax.legend(frameon=False,loc='upper right',fontsize=8)
ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.savefig(FIG/'outcomes.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(6.5,2.7),gridspec_kw={'width_ratios':[1.15,1]})
ax=axes[0];ys=np.arange(4)
for y,row in zip(ys,rows[1:]):
    ax.plot([row['base_finish'],row['r009_finish']],[y,y],color='#b6c3cc',lw=3,zorder=1)
ax.scatter([r['base_finish'] for r in rows[1:]],ys,color=grey,s=35,label='База',zorder=3)
ax.scatter([r['r009_finish'] for r in rows[1:]],ys,color=blue,s=35,label='r009',zorder=3)
ax.set_yticks(ys,labels[1:]);ax.invert_yaxis();ax.set_xlim(0,26);ax.set_xticks([0,5,10,15,20,25])
ax.set_xlabel('Финишировало из 25');ax.set_title('(а) Возмущённые старты',loc='left')
ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower left',fontsize=8)
ax=axes[1]
for pol,col,lab in [(old,grey,'База'),(new,blue,'r009')]:
    times=sorted(t['time_s'] for t in pol['trials'] if t['reason']=='finish')
    ax.step([0]+times+[60],[0]+[100*(i+1)/101 for i in range(len(times))]+[100*len(times)/101],where='post',color=col,label=lab,lw=1.8)
ax.set_xlim(35,60);ax.set_ylim(0,100);ax.set_xlabel('Физическое время, с');ax.set_ylabel('Финишировавшие, % от 101')
ax.set_title('(б) Завершение дистанции',loc='left');ax.grid(alpha=.15)
fig.tight_layout(w_pad=1.4);fig.savefig(FIG/'robustness.pdf');plt.close(fig)

motion=json.loads((DATA/'r009_motion.json').read_text())
t=np.array(motion['phase_s']);q=np.array(motion['joint_target_rad'])
fig,axes=plt.subplots(1,2,figsize=(6.5,2.35),sharey=True)
for j,ax in enumerate(axes):
    for k,label,color in [(0,'Тазобедренный',blue),(1,'Коленный',red)]:
        ax.plot(t,q[:,j*2+k],'-o',color=color,ms=3,lw=1.5,label=label)
    ax.set_xlabel('Время от запуска движения, с');ax.set_title('Левая нога' if j==0 else 'Правая нога',loc='left')
    ax.set_ylim(-2.9,2.9);ax.set_xticks([0,1,2,3,4,5,6]);ax.grid(alpha=.15)
axes[0].set_ylabel('Целевой угол, рад');axes[0].legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout();fig.savefig(FIG/'joint_targets.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,2.65));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,label,color='#f1f4f6'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.06',lw=.8,ec='#738695',fc=color))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.7)
def arrow(x0,y0,x1,y1):ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops={'arrowstyle':'->','color':'#526a7c','lw':1})
ax.text(0,3.75,'ОБУЧЕНИЕ И ПРОВЕРКА',fontsize=8,color=blue,weight='bold')
box(.1,2.35,2.1,1,'Поиск учителя\nCEM + физика\nPython / Rust')
box(2.85,2.35,2.6,1,'Ресурсное STDP\nKNP 2.0.0\nCPU backend, C++','#e8f0f5')
box(6.05,2.35,3.75,1,'Экспорт весов\nПарная проверка и принятие\nPython / Rust')
arrow(2.23,2.85,2.75,2.85);arrow(5.5,2.85,5.97,2.85)
ax.text(0,1.8,'ИСПОЛНЕНИЕ В SIMROKI',fontsize=8,color=blue,weight='bold')
box(.1,.25,2.1,1,'Датчики\nскорость, угол, ω,\nвысота, контакты')
box(2.85,.25,2.6,1,'Карты KNP\nНаклон, фаза, подъём\n1108 обученных весов','#e8f0f5')
box(6.05,.25,3.75,1,'Эталоны RABBIT и PD\n4 суставных привода\nRapier, Δt = 1/240 с')
arrow(2.23,.75,2.75,.75);arrow(5.5,.75,5.97,.75)
ax.plot([7.9,7.9,4.15],[2.27,1.6,1.6],color='#526a7c',lw=1)
arrow(4.15,1.6,4.15,1.3)
fig.savefig(FIG/'architecture.pdf');plt.close(fig)

manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [DATA, ROOT/'code'] for p in sorted(folder.rglob('*')) if p.is_file()}
(ROOT/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:results[k] for k in ['n','gained_finishes','lost_finishes','mean_return_gain','paired_return_gain_lower95_one_sided']},ensure_ascii=False))
