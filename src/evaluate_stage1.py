import argparse,json,math,gc,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from custom_models.drope import DroPELlamaForCausalLM,DroPELlamaConfig
from train_stage1 import ce
from common import dump

@torch.inference_mode()
def score(model,blocks,length):
    full=[];tail=[]
    for block in blocks:
        ids=torch.tensor(np.asarray(block[:length],dtype=np.int64),device='cuda').unsqueeze(0)
        logits=model(input_ids=ids,use_cache=False).logits
        full.append(ce(logits[:,:-1],ids[:,1:]).item())
        # Same target positions are compared for every model at this length.
        start=max(0,length-512)
        tail.append(ce(logits[:,start:-1],ids[:,start+1:]).item())
        del logits,ids
    mean=sum(full)/len(full)
    return {'ce':mean,'ppl':math.exp(mean),'tail512_ce':sum(tail)/len(tail),'n_blocks':len(blocks),'n_tokens_scored':len(blocks)*(length-1)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=json.loads(Path(p.parse_args().config).read_text())
    root=Path(c['night_dir']);m=json.loads((root/'checkpoint_metadata.json').read_text())
    blocks=np.load(root/'cache/test.npy');rows=[]
    specs=[('RoPE_base',c['checkpoint']),('QKNoPE_step0',c['checkpoint']),('DroPE_final',m['hf_checkpoint'])]
    for name,path in specs:
        if name=='RoPE_base': model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16,attn_implementation='sdpa')
        elif name=='QKNoPE_step0':
            cfg=DroPELlamaConfig.from_pretrained(path);cfg.attention_type='qk_norm_nope'
            model=DroPELlamaForCausalLM.from_pretrained(path,config=cfg,dtype=torch.float16,attn_implementation='sdpa')
        else:model=DroPELlamaForCausalLM.from_pretrained(path,dtype=torch.float16,attn_implementation='sdpa')
        model.to('cuda').eval()
        variants=[False,True] if name=='DroPE_final' else [False]
        for scaled in variants:
            for length in [2048,4096,8192]:
                beta=1+0.103*math.log(length/2048) if scaled else 1.0
                for layer in model.model.layers: layer.self_attn.scaling=(model.config.hidden_size/model.config.num_attention_heads)**-0.5*beta
                row={'model':name,'length':length,'attention_scaled':scaled,'beta':beta,'step':m['step'] if name=='DroPE_final' else 0}
                try:row.update(score(model,blocks,length),status='OK')
                except torch.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache();row.update(status='OOM')
                rows.append(row);dump(root/'test_results.json',rows);print('TEST',json.dumps(row),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    import csv
    keys=sorted({k for r in rows for k in r})
    with (root/'test_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        log=json.loads((root/'training_log.json').read_text())
        fig,ax=plt.subplots(figsize=(9,4));ax.plot([r['step'] for r in log],[r['loss'] for r in log],alpha=.6,label='train CE')
        curve=json.loads((root/'dev_curve.json').read_text()) if (root/'dev_curve.json').exists() else []
        if curve:ax.plot([r['step'] for r in curve],[r['dev_loss'] for r in curve],'o-',label='dev CE')
        ax.set(xlabel='Optimizer step',ylabel='Cross entropy');ax.legend();fig.tight_layout();fig.savefig(root/'learning_curve.png',dpi=160);plt.close(fig)
    except ImportError:pass
if __name__=='__main__':main()
