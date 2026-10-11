"""CPU-only analysis of saved DroPE metrics. Never loads a model or starts training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(root):
    root = Path(root)
    before = root/'results/stage1/evaluation'
    after = root/'results/evaluation'
    assert read(before/'check_config.json')['test_sha256'] == read(after/'check_config.json')['test_sha256'], 'Different test data'
    for folder in (before, after):
        assert all(v == 0 for v in read(folder/'worker_status.json').values()), 'Incomplete evaluation'
        assert not read(folder/'drope/integrity.json')['suspect'], 'Integrity check flagged'
    return root, before, after


def tables(root):
    root, before, after = load(root)
    summaries = []
    for stage, folder in [('before', before), ('after', after)]:
        df = pd.read_csv(folder/'lm_summary.csv');df['stage'] = stage;summaries.append(df)
    lm = pd.concat(summaries, ignore_index=True)
    selected = lm[((lm['length'] == 2048) & (lm['variant'] == 'unscaled')) | ((lm['length'] > 2048) & ((lm['variant'] == 'dev_tuned') | (lm['model'] == 'RoPE_base')))].copy()
    selected['label'] = np.where(selected['model'] == 'RoPE_base', 'RoPE base', 'DroPE '+selected['stage'])
    comparison = selected.drop_duplicates(['label', 'length']).pivot(index='length', columns='label', values='ppl')
    comparison['DroPE PPL reduction, %'] = (1-comparison['DroPE after']/comparison['DroPE before'])*100
    runs = []
    for stage in ['stage1', 'stage2']:
        r = read(root/f'results/{stage}/training_result.json')
        runs.append({'stage':stage, **r})
    resources = pd.DataFrame(runs)
    return lm, selected, comparison, resources


def save(fig, root, name):
    out = Path(root)/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def build(root):
    root, before, after = load(root)
    lm, selected, comparison, resources = tables(root)
    out = root/'results/tables';out.mkdir(parents=True,exist_ok=True)
    for name, frame in [('lm_all_variants',lm),('lm_selected',selected),('ppl_comparison',comparison),('resources',resources)]:
        frame.to_csv(out/(name+'.csv'), index=name=='ppl_comparison')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    colors={'RoPE base':'#646b73','DroPE before':'#d28333','DroPE after':'#167d9a'}
    # Combine absolute steps, retaining evaluated step 8712 from stage2.
    curves = pd.concat([pd.DataFrame(read(root/f'results/{s}/dev_curve.json')) for s in ['stage1','stage2']]).drop_duplicates('step',keep='last').sort_values('step')
    curves.to_csv(out/'dev_curve_combined.csv',index=False)
    train = pd.concat([pd.DataFrame(read(root/f'results/{s}/training_log.json')) for s in ['stage1','stage2']]).sort_values('step')
    assert np.isfinite(train['loss']).all() and not train['step'].duplicated().any()
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(train['step'],train['loss'].rolling(200,min_periods=1).mean(),color=colors['DroPE after']);axes[0].set(xlabel='Absolute step',ylabel='Train CE (200-step moving mean)',title='Training recovery')
    axes[1].plot(curves['step'],curves['dev_loss'],'o-',color=colors['DroPE after']);axes[1].set(xlabel='Absolute step',ylabel='Dev CE',title='Held-out language-model loss')
    for ax in axes:ax.axvline(8712,ls='--',color='#888',label='Resume: step 8712');ax.legend()
    fig.tight_layout();save(fig,root,'01_learning_curves')
    fig, ax = plt.subplots(figsize=(7,4))
    for label,g in selected.groupby('label'):
        g=g.drop_duplicates('length').sort_values('length');ax.plot(g['length'],g['ppl'],'o-',label=label,color=colors[label])
    ax.set(xlabel='Context tokens',ylabel='Test PPL (log scale)',title='Long-context LM: beta selected on dev',yscale='log',xticks=[2048,4096,8192]);ax.legend();save(fig,root,'02_context_ppl')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,(stage,folder) in zip(axes,[('before',before),('after',after)]):
        d=pd.read_csv(folder/'lm_summary.csv');d=d[d['model']!='RoPE_base']
        for variant,g in d.groupby('variant'):
            g=g.sort_values('length');ax.plot(g['length'],g['ppl'],'o-',label=variant)
        ax.set(title='DroPE '+stage,xlabel='Context tokens',ylabel='Test PPL',yscale='log',xticks=[2048,4096,8192]);ax.legend()
    fig.tight_layout();save(fig,root,'03_scaling_test')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,length in zip(axes,[4096,8192]):
        for stage,folder in [('before',before),('after',after)]:
            d=pd.DataFrame(read(folder/'drope/scaling_sweep.json'));g=d[(d['length']==length)&(d['status']=='OK')].sort_values('beta')
            ax.plot(g['beta'],g['dev_ce'],'o-',label='DroPE '+stage,color=colors['DroPE '+stage])
            best=g.loc[g['dev_ce'].idxmin()];ax.scatter([best['beta']],[best['dev_ce']],s=100,facecolors='none',edgecolors=colors['DroPE '+stage])
        ax.set(title=f'Dev beta sweep: {length} tokens',xlabel='Attention beta',ylabel='Dev CE');ax.legend()
    fig.tight_layout();save(fig,root,'04_beta_dev')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,length in zip(axes,[4096,8192]):
        for stage,folder in [('before',before),('after',after)]:
            d=pd.read_csv(folder/'position_losses.csv');d=d[(d['length']==length)&(((d['model']=='RoPE_base')&(d['variant']=='unscaled'))|((d['model']!='RoPE_base')&(d['variant']=='dev_tuned')))]
            for model,g in d.groupby('model'):
                if model=='RoPE_base' and stage=='before':continue
                label='RoPE base' if model=='RoPE_base' else 'DroPE '+stage
                m=g.groupby('target_end')['ce'].mean();ax.plot(m.index,m.values,label=label,color=colors[label])
        ax.set(title=f'Loss by position: context {length}',xlabel='Target token position (512-token segment end)',ylabel='Mean segment CE');ax.legend()
    fig.tight_layout();save(fig,root,'05_position_loss')
    # Diagnostic sample counts are displayed, including true zero cells.
    d=pd.read_csv(after/'niah_summary.csv');d=d[d['kind']=='main']
    chosen=d[((d['model']=='RoPE_base')&(d['variant']=='unscaled'))|((d['model']!='RoPE_base')&(((d['length']==2048)&(d['variant']=='unscaled'))|((d['length']>2048)&(d['variant']=='dev_tuned'))))]
    chosen.to_csv(out/'niah_selected.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(model,g) in zip(axes,chosen.groupby('model',sort=False)):
        v=g.pivot(index='depth',columns='length',values='accuracy').sort_index();im=ax.imshow(v.values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=range(len(v.columns)),xticklabels=v.columns,yticks=range(len(v.index)),yticklabels=v.index,xlabel='Context tokens',ylabel='Needle depth',title=model)
        for i,depth in enumerate(v.index):
            for j,length in enumerate(v.columns):
                row=g[(g['depth']==depth)&(g['length']==length)].iloc[0]
                ax.text(j,i,f"{int(row['correct'])}/{int(row['n'])}",ha='center',va='center',color='white' if row['accuracy']>.6 else 'black')
        ax.grid(False)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Exact-match accuracy');save(fig,root,'06_niah')
    # Additional run was recovered from console, not a surviving raw archive.
    extra = pd.read_csv(root/'results/extra/lm_summary.csv')
    final = selected[selected['label']=='DroPE after'].copy()
    final['method']='DroPE (dev beta)';final['provenance']='saved_machine_metrics'
    expanded = pd.concat([extra.assign(method=extra['method'].replace({'default':'RoPE base','yarn':'YaRN','dynamic':'Dynamic NTK'})), final],ignore_index=True)
    expanded.to_csv(out/'lm_with_baselines.csv',index=False)
    fig, ax=plt.subplots(figsize=(8,4.6))
    for label,g in expanded.groupby('method'):
        g=g.sort_values('length');ax.plot(g['length'],g['ppl'],'o-',label=label)
    ax.set(xlabel='Context tokens',ylabel='Test PPL (log scale)',yscale='log',xticks=[2048,4096,8192],title='LM controls: 8 fixed test blocks')
    ax.legend();save(fig,root,'07_baseline_controls')
    niah=pd.read_csv(root/'results/extra/niah_summary.csv')
    chosen=niah[niah['kind'].isin(['short_control','main'])].copy()
    chosen.to_csv(out/'niah_expanded.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.6))
    labels=['512 / depth .9','2048 / depth .1','2048 / depth .5','2048 / depth .9']
    for j,model in enumerate(['rope','drope']):
        d=chosen[chosen.model==model].sort_values(['length','depth'])
        bars=ax.bar(np.arange(4)+(j-.5)*.32,d['accuracy'],width=.32,label=model)
        for bar,(_,row) in zip(bars,d.iterrows()):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f"{int(row.correct)}/{int(row.n)}",ha='center')
    ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1.15),ylabel='Exact-match accuracy',title='Expanded NIAH: recovered console counts');ax.legend();save(fig,root,'08_niah_expanded')
    return comparison

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    print(build(a.root).round(3));print('Saved figures/*.png, figures/*.pdf and results/tables/*.csv')
