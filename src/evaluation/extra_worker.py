"""Additional fixed-template NIAH + fixed-factor RoPE baselines (Linux/CUDA)."""
import sys, json, time, re, gc, hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_rope_utils import rope_config_validation
from worker import fix_cache_keyword, integrity, token_losses, make_prompt
from custom_models.drope import DroPELlamaForCausalLM

class BudgetEnd(Exception): pass

@torch.inference_mode()
def main():
    c=json.loads(Path(sys.argv[1]).read_text());role=sys.argv[2];out=Path(c['out'])
    rows=[];lm=[];events=[]
    def budget():
        if time.time()>=c['deadline']:raise BudgetEnd()
    def store():
        for name,value in [('niah',rows),('lm',lm),('events',events)]:
            p=out/f'{role}_{name}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
    try:
        budget();test=np.load(c['test']);tok=AutoTokenizer.from_pretrained(c['base'])
        cls=AutoModelForCausalLM if role=='rope' else DroPELlamaForCausalLM
        model=cls.from_pretrained(c['base'] if role=='rope' else c['hf'],dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
        if role=='drope':fix_cache_keyword(model)
        checks=integrity(model,test[0]);(out/f'{role}_integrity.json').write_text(json.dumps(checks,indent=2))
        assert not checks['suspect'],f'Integrity failed: {checks}'
        print(role,'INTEGRITY OK',flush=True)
        conditions=[(512,.9,s,True,'short_control') for s in range(12)]
        conditions += [(512,.9,s,False,'no_needle_control') for s in range(4)]
        conditions += [(2048,d,s,True,'main') for d in [.1,.5,.9] for s in range(8)]
        for length,depth,s,needle,kind in conditions:
            budget();ids,number,key,pos=make_prompt(tok,test[s%len(test)],length,depth,s,needle)
            tensor=torch.tensor([ids],device='cuda')
            output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),max_new_tokens=24,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,max_time=max(1,min(30,c['deadline']-time.time())))
            text=tok.decode(output[0,len(ids):],skip_special_tokens=True);numbers=re.findall(r'(?<!\d)\d{7}(?!\d)',text)
            rows.append(dict(model=role,kind=kind,length=length,depth=depth,sample=s,seed=20261010+s,key=key,expected=number,needle_present=needle,needle_token_position=pos if needle else None,generated=text,extracted_numbers=numbers,correct=bool(needle and numbers and numbers[0]==number),expected_number_generated=bool(numbers and numbers[0]==number),prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int32).tobytes()).hexdigest()))
            store();print(role,'NIAH',kind,length,depth,s,rows[-1]['correct'],flush=True);del tensor,output
        if role=='rope':
            del model;gc.collect();torch.cuda.empty_cache()
            orig=AutoConfig.from_pretrained(c['base']).max_position_embeddings;assert orig==2048
            jobs=[('default',l) for l in [2048,4096,8192]]+[(m,l) for m in ['yarn','dynamic'] for l in [4096,8192]]
            for method,length in jobs:
                budget();scaling=None if method=='default' else {'rope_type':method,'factor':float(length/orig)}
                cfg=AutoConfig.from_pretrained(c['base'])
                if method=='yarn':
                    scaling['original_max_position_embeddings']=orig
                    # Metadata fix relative to the console-recovered run. Keep dynamic's original limit.
                    cfg.max_position_embeddings=length
                cfg.rope_scaling=scaling;rope_config_validation(cfg)
                model=AutoModelForCausalLM.from_pretrained(c['base'],config=cfg,dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
                assert model.model.rotary_emb.rope_type==method
                for sample,block in enumerate(test):
                    budget();ids=torch.tensor(np.array(block[:length],dtype=np.int64),device='cuda').unsqueeze(0)
                    logits=model(input_ids=ids,use_cache=False).logits;loss=token_losses(logits,ids)
                    lm.append(dict(model='RoPE_base',method=method,length=length,factor=1.0 if scaling is None else scaling['factor'],block=sample,ce=float(loss.mean()),tail512_ce=float(loss[-512:].mean()),rope_scaling=scaling))
                    store();print(role,'LM',method,length,sample,lm[-1]['ce'],flush=True);del ids,logits
                del model;gc.collect();torch.cuda.empty_cache()
    except BudgetEnd:events.append(dict(status='TIMEOUT',message='Partial results retained'))
    except Exception as exc:
        events.append(dict(status='ERROR',type=type(exc).__name__,message=str(exc)));raise
    finally:store()

if __name__=='__main__':main()
