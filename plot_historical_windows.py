"""Report plots for three paired historical windows."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path(__file__).resolve().parent/'historical-window-results'
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family']=next(f for f in ('Hiragino Sans GB','PingFang SC','Noto Sans CJK SC','SimHei') if f in fonts)
with (out/'region-size.csv').open() as f:
    sizes=list(csv.DictReader(f))
records=json.loads((out/'summary.json').read_text())['results']
counts=[4,16,32,64]
sides=[1,2,4,5,8,10,20,40]
assert len(sizes)==120 and len(records)==45
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for n,marker in zip(counts,['o','s','^','x']):
    for ax,key,title,mult in zip(axes,['mean_error_km','leaked_bits','delivery'],['平均定位误差（千米）','位置与观测的互信息（比特）','送达率（%）'],[1,1,100]):
        y=[np.mean([float(r[key]) for r in sizes if int(r['side_km'])==s and int(r['messages'])==n])*mult for s in sides]
        ax.plot(sides,y,marker=marker,fillstyle='none',label=f'{n} 条')
        ax.set(xlabel='区域边长（千米）',ylabel=title)
axes[0].legend();fig.savefig(out/'size-windows.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for mode,label in [('fixed_region','不加差分隐私'),('random_reuse','加入差分隐私')]:
    groups=[[r for r in records if r['mode']==mode and r['messages']==n] for n in counts]
    assert all(len(g)==3 for g in groups)
    for ax,values,title in zip(axes,[[np.mean([r['mean_error_km'] for r in g]) for g in groups],[100*np.mean([r['delivery_rate'] for r in g]) for g in groups]],['平均定位误差（千米）','送达率（%）']):
        ax.plot(counts,values,marker=marker,label=label)
        ax.set(xticks=counts,xlabel='累计消息数',ylabel=title)
axes[0].legend();fig.savefig(out/'dp-windows.png',dpi=180)
