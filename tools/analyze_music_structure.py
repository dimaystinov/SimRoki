"""Analyze the local bundled MP3. Requires ffmpeg, numpy, scipy, matplotlib.
Run from repository root: python3 tools/analyze_music_structure.py
"""
import json, subprocess
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('target/music-analysis');out.mkdir(parents=True,exist_ok=True)
sr=22050;hop=220
raw=subprocess.check_output(['ffmpeg','-v','error','-i','assets/audio/why-not.mp3','-f','f32le','-ac','1','-ar',str(sr),'pipe:1'])
y=np.frombuffer(raw,dtype='<f4')
f,t,z=signal.stft(y,sr,nperseg=2048,noverlap=2048-hop)
a=np.abs(z)
flux=np.maximum(np.diff(np.log1p(100*a),axis=1),0).sum(axis=0)
flux=np.maximum(flux-signal.medfilt(flux,101),0);ft=t[1:]
# Positive spectral flux is an onset proxy, not a semantic bar detector.
ac=signal.fftconvolve(flux,flux[::-1],mode='full')[len(flux)-1:]
ac/=np.arange(len(flux),0,-1);ac/=ac[0];lags=np.arange(len(ac))*hop/sr
# Timbre/pitch/energy summary every quarter second. No learned beat/bar model.
bands=np.geomspace(40,10000,49)
spec=np.array([np.log1p(100*a[(f>=lo)&(f<hi)].mean(axis=0)) for lo,hi in zip(bands[:-1],bands[1:]) if np.any((f>=lo)&(f<hi))])
bin_t=np.arange(0,len(y)/sr-.25,.25)
features=np.array([spec[:,(t>=x)&(t<x+.25)].mean(axis=1) for x in bin_t])
features=(features-features.mean(axis=0))/(features.std(axis=0)+1e-8)
# Remove the introduction/outro; compare complete sequences, rather than a few selected hits.
mask=(bin_t>=8)&(bin_t<152);v=features[mask]
repeat=[]
for lag in np.arange(.25,32.01,.25):
 k=round(lag/.25); left=v[:-k].ravel();right=v[k:].ravel()
 repeat.append(float(np.corrcoef(left,right)[0,1]))
rep_lags=np.arange(.25,32.01,.25)
# Compare onset strength by candidate phrase phase; loud hits alone do not prove bar boundaries.
phase_results={}
for period in [2,4,8,16]:
 phases=[]
 for offset in np.arange(0,period,.25):
  centers=np.arange(8+offset,152,period)
  peaks=[float(flux[(ft>=c-.04)&(ft<=c+.08)].max()) for c in centers]
  phases.append((float(np.mean(peaks)),float(offset)))
 phase_results[str(period)]=sorted(phases,reverse=True)[:3]
summary={
 'decoded_duration_s':len(y)/sr,
 'onset_autocorrelation':{str(x):float(ac[np.argmin(abs(lags-x))]) for x in [.25,.5,1,2,4,8,16]},
 'spectral_sequence_correlation':{str(x):repeat[round(x/.25)-1] for x in [1,2,4,8,16,32]},
 'strongest_repeat_lags':[(float(rep_lags[i]),repeat[i]) for i in np.argsort(repeat)[-10:][::-1]],
 'candidate_phase_peak_strength':phase_results,
 'frequency_bands_used':len(spec),
 'method':'mono 22050 Hz; 2048-sample STFT / 220-sample hop; positive log spectral flux; nonempty log-frequency bands averaged in 0.25 s bins; Pearson sequence correlation over 8–152 s',
 'limitation':'Signal periodicity does not uniquely identify musical metre or the first beat of a bar. 8 s markers are candidate blocks, not automatically confirmed downbeats.'}
(out/'analysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(4,1,figsize=(14,13),layout='constrained')
fig.suptitle('GHOSTFACE PLAYA — Why Not: measured repetition, not assumed bars',fontsize=17,weight='bold')
# Envelope overview
win=2205;n=len(y)//win;rms=np.sqrt((y[:n*win].reshape(n,win)**2).mean(axis=1));rt=(np.arange(n)+.5)*.1
axes[0].plot(rt,rms,color='#304878',lw=1)
for x in np.arange(0,166,8):axes[0].axvline(x,color='#b64d71',alpha=.3,lw=.8)
axes[0].set(title='1. Whole recording: RMS loudness; pink lines = proposed 8-second cuts',xlabel='Time (s)',ylabel='RMS amplitude',xlim=(0,165))
sel=(ft>=8)&(ft<=24)
axes[1].plot(ft[sel],flux[sel],color='#304878',lw=1)
for x in np.arange(8,24.1,2):axes[1].axvline(x,color='#999999',alpha=.45,lw=.8)
for x in [8,16,24]:axes[1].axvline(x,color='#b64d71',lw=2)
axes[1].set(title='2. Onset strength, 8–24 s: regular hits; pink = 8 s, grey = 2 s',xlabel='Time (s)',ylabel='Positive spectral flux',xlim=(8,24))
axes[2].plot(lags,ac,color='#317765')
for x in [.25,.5,1,2,4,8]:
 idx=np.argmin(abs(lags-x));axes[2].plot(lags[idx],ac[idx],'o',color='#b64d71');axes[2].annotate(f'{x:g}s',(lags[idx],ac[idx]),xytext=(3,7),textcoords='offset points')
axes[2].set(title='3. Onset autocorrelation: how regularly the rhythmic pattern repeats',xlabel='Lag (s)',ylabel='Normalized correlation',xlim=(.1,8.2),ylim=(0,1.08))
axes[3].plot(rep_lags,repeat,color='#7560a4')
for x in [2,4,8,16,32]:
 val=repeat[round(x/.25)-1];axes[3].plot(x,val,'o',color='#b64d71');axes[3].annotate(f'{x}s: {val:.2f}',(x,val),xytext=(-25,9),textcoords='offset points')
axes[3].set(title='4. Spectral-pattern repetition, central section 8–152 s',xlabel='Lag (s)',ylabel='Pearson correlation',xlim=(0,33),ylim=(-.2,1))
fig.savefig(out/'why-not-structure.png',dpi=150)
