import argparse,json,os,math,time,random,re,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from custom_models.drope import DroPELlamaForCausalLM
from common import dump

class BudgetEnd(Exception):pass

def cache_keyword_adapter(original, *args, **kwargs):
    # Transformers 4.56 LlamaDecoderLayer passes plural past_key_values.
    # Sakana's pinned custom attention consumes singular past_key_value.
    plural=kwargs.pop('past_key_values',None)
    if kwargs.get('past_key_value') is None and plural is not None:
        kwargs['past_key_value']=plural
    return original(*args,**kwargs)

def fix_cache_keyword(model):
    from functools import partial
    for layer in model.model.layers:
        attn=layer.self_attn
        if hasattr(attn,'q_norm') and not getattr(attn,'_cache_keyword_fixed',False):
            attn.forward=partial(cache_keyword_adapter,attn.forward)
            attn._cache_keyword_fixed=True

def beta_paper(length):return 1+0.103*math.log(max(1,length/2048))

def set_beta(model,beta):
    for layer in model.model.layers:layer.self_attn.scaling=layer.self_attn.head_dim**-0.5*beta

@torch.inference_mode()
def token_losses(logits,ids):
    parts=[]
    for i in range(0,ids.shape[1]-1,128):
        a=logits[:,i:min(i+128,ids.shape[1]-1)].float()
        b=ids[:,i+1:min(i+129,ids.shape[1])]
        parts.append(F.cross_entropy(a.reshape(-1,a.shape[-1]),b.reshape(-1),reduction='none').cpu())
    return torch.cat(parts).numpy()

def make_prompt(tok,source,length,depth,sample,needle=True):
    rng=random.Random(20261010+sample)
    key=['blue-lantern','quiet-river'][sample%2];number=str(rng.randint(1000000,9999999))
    header=tok.encode('Read the following text and remember the special magic number.\n\n',add_special_tokens=False)
    query=tok.encode(f'\n\nThe special magic number for {key} mentioned in the text is:',add_special_tokens=False)
    fact=tok.encode(f'\n\nThe special magic number for {key} is {number}.\n\n',add_special_tokens=False) if needle else []
    count=length-len(header)-len(query)-len(fact)
    assert count>0 and len(source)>=count
    cut=round(depth*count)
    ids=header+list(map(int,source[:cut]))+fact+list(map(int,source[cut:count]))+query
    assert len(ids)==length
    return ids,number,key,len(header)+cut

@torch.inference_mode()
def integrity(model,block):
    set_beta(model,1)
    ids=torch.tensor(np.array(block[:512],dtype=np.int64),device='cuda').unsqueeze(0)
    prefix=model(input_ids=ids[:,:256],use_cache=False).logits.float()
    full=model(input_ids=ids,use_cache=False).logits[:,:256].float()
    causal=float((prefix-full).abs().mean());causalmax=float((prefix-full).abs().max())
    sdpa=model(input_ids=ids,use_cache=False).logits.float()
    # Switch both model/mask selection and attention modules, avoiding stale backend state.
    model.set_attn_implementation('eager')
    eager=model(input_ids=ids,use_cache=False).logits.float()
    mean=float((sdpa-eager).abs().mean());maximum=float((sdpa-eager).abs().max())
    ce_sdpa=float(token_losses(sdpa,ids).mean());ce_eager=float(token_losses(eager,ids).mean())
    model.set_attn_implementation('sdpa')
    cached=model(input_ids=ids[:,:256],use_cache=True)
    continuation=model(input_ids=ids[:,256:],past_key_values=cached.past_key_values,use_cache=True).logits.float()
    cache_delta=float((sdpa[:,256:]-continuation).abs().mean())
    return {'prefix_mean_abs_logit_delta':causal,'prefix_max_abs_logit_delta':causalmax,
            'cache_mean_abs_logit_delta':cache_delta,'sdpa_eager_mean_abs_logit_delta':mean,'sdpa_eager_max_abs_logit_delta':maximum,
            'sdpa_ce':ce_sdpa,'eager_ce':ce_eager,'ce_delta':abs(ce_sdpa-ce_eager),
            'suspect':causal>.05 or cache_delta>.05 or abs(ce_sdpa-ce_eager)>.05,
            'note':'FP16 numerical diagnostic; threshold flags investigation, not a formal proof.'}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--model',choices=['rope','drope'],required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());root=Path(c['out_dir']);out=root/a.model;out.mkdir(exist_ok=True)
    name='RoPE_base' if a.model=='rope' else f"DroPE_step{c['step']}"
    end=c['worker_deadline'];events=[];profiles=[];niah=[];sweep=[];tuned={}
    def budget():
        if time.time()>=end:raise BudgetEnd()
    def store():
        dump(out/'profile.json',profiles);dump(out/'niah.json',niah);dump(out/'scaling_sweep.json',sweep);dump(out/'events.json',events)
    def error(task,exc):
        events.append({'task':task,'error':type(exc).__name__,'message':str(exc)[:1000]});store();gc.collect();torch.cuda.empty_cache()
    tok=AutoTokenizer.from_pretrained(c['base_path'])
    path=c['base_path'] if a.model=='rope' else c['hf_path']
    cls=AutoModelForCausalLM if a.model=='rope' else DroPELlamaForCausalLM
    model=cls.from_pretrained(path,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    if a.model=='drope':
        assert model.config.attention_type=='qk_norm_nope'
        assert all(type(x.self_attn).__name__=='QKNormNoPELlamaAttention' for x in model.model.layers)
    dump(out/'model_checks.json',{'model':name,'step':0 if a.model=='rope' else c['step'],
          'attention_type':getattr(model.config,'attention_type','rope'),'training':model.training,
          'head_dim':model.model.layers[0].self_attn.head_dim,'q_norm_present':hasattr(model.model.layers[0].self_attn,'q_norm'),
          'norm_eps':getattr(model.model.layers[0].self_attn,'q_norm',model.model.layers[0].self_attn).__dict__.get('eps')})
    test=np.load(c['test_path']);dev=np.load(c['dev_path'])
    try:
        budget()
        try:
            if a.model=='drope':
                before=integrity(model,test[0]);dump(out/'integrity_before_cache_fix.json',before)
                fix_cache_keyword(model)
                print(name,'CACHE KEYWORD ADAPTER applied; model weights unchanged',flush=True)
            checks=integrity(model,test[0]);dump(out/'integrity.json',checks);print(name,'INTEGRITY',checks,flush=True)
            if checks['cache_mean_abs_logit_delta']>.05:
                raise RuntimeError('KV cache still inconsistent after adapter: do not interpret cached NIAH')
        except (RuntimeError,AttributeError) as exc:
            error('integrity',exc)
            raise
        # Fit beta on DEV only, never on test; dev blocks are contiguous packed tokens.
        if a.model=='drope':
            dev_long=dev.reshape(-1)[:16384].reshape(2,8192)
            for length in [4096,8192]:
                betas=sorted(set([1.0,beta_paper(length),1.25,1.5,1.75,2.0]))
                for beta in betas:
                    budget();set_beta(model,beta);values=[]
                    try:
                        for block in dev_long:
                            budget();ids=torch.tensor(np.array(block[:length],dtype=np.int64),device='cuda').unsqueeze(0)
                            logits=model(input_ids=ids,use_cache=False).logits
                            values.append(float(token_losses(logits,ids).mean()));del logits,ids
                        row={'length':length,'beta':beta,'dev_ce':sum(values)/len(values),'dev_blocks':len(values),'status':'OK'}
                        sweep.append(row);store();print(name,'BETA',row,flush=True)
                    except torch.OutOfMemoryError as exc:error(f'beta_{length}_{beta}',exc)
                available=[r for r in sweep if r['length']==length and r['status']=='OK']
                if available:tuned[str(length)]=min(available,key=lambda r:r['dev_ce'])['beta']
            dump(out/'selected_beta.json',tuned)
        # Reproduce all 8 test blocks; retain per-block metrics and positional CE.
        for length in [2048,4096,8192]:
            variants={'unscaled':1.0}
            if a.model=='drope' and length>2048:
                variants.update(paper=beta_paper(length),dev_tuned=tuned.get(str(length),beta_paper(length)))
            for variant,beta in variants.items():
                for sample,block in enumerate(test[:c['test_blocks']]):
                    budget();set_beta(model,beta)
                    try:
                        ids=torch.tensor(np.array(block[:length],dtype=np.int64),device='cuda').unsqueeze(0)
                        logits=model(input_ids=ids,use_cache=False).logits;losses=token_losses(logits,ids)
                        row={'model':name,'length':length,'variant':variant,'beta':beta,'block':sample,
                             'ce':float(losses.mean()),'tail512_ce':float(losses[-512:].mean()),'status':'OK',
                             'segments':[{'target_start':start+1,'target_end':min(start+512,length-1),
                                          'ce':float(losses[start:start+512].mean())} for start in range(0,len(losses),512)]}
                        profiles.append(row);store();del logits,ids
                    except torch.OutOfMemoryError as exc:error(f'profile_{length}_{variant}_{sample}',exc)
                print(name,'PROFILE',length,variant,'complete',flush=True)
        # Greedy base-LM completion, not chat template. Identical prompts/seeds across models.
        conditions=[(512,.9,s,True,'short_control') for s in range(2)]
        conditions += [(512,.9,s,False,'no_needle_control') for s in range(2)]
        conditions += [(length,depth,s,True,'main') for length in [2048,4096,8192] for depth in [.1,.5,.9] for s in range(c['niah_samples'])]
        for length,depth,sample,needle,kind in conditions:
            variants={'unscaled':1.0}
            if a.model=='drope' and length>2048:variants.update(paper=beta_paper(length),dev_tuned=tuned.get(str(length),beta_paper(length)))
            for variant,beta in variants.items():
                budget();set_beta(model,beta)
                ids,expected,key,position=make_prompt(tok,test[sample%len(test)],length,depth,sample,needle)
                try:
                    tensor=torch.tensor([ids],device='cuda')
                    # max_time supplements wall-budget check during decoding; prefill is not interruptible here.
                    output=model.generate(tensor,max_new_tokens=24,do_sample=False,use_cache=True,
                             pad_token_id=tok.eos_token_id,max_time=max(1,min(90,end-time.time())))
                    text=tok.decode(output[0,len(ids):],skip_special_tokens=True)
                    numbers=re.findall(r'(?<!\d)\d{7}(?!\d)',text)
                    row={'model':name,'kind':kind,'length':length,'depth':depth,'sample':sample,
                         'variant':variant,'beta':beta,'key':key,'expected':expected,'needle_present':needle,
                         'needle_token_position':position if needle else None,'generated':text,
                         'extracted_numbers':numbers,'correct':bool(numbers and numbers[0]==expected),
                         'status':'OK','prompt_tokens':len(ids)}
                    niah.append(row);store();print(name,'NIAH',kind,length,depth,variant,row['correct'],repr(text),flush=True)
                    del output,tensor
                except torch.OutOfMemoryError as exc:error(f'niah_{length}_{variant}',exc)
    except BudgetEnd:
        events.append({'task':'budget','message':'Worker deadline reached; partial results retained'})
    finally:
        store();dump(out/'finished.json',{'model':name,'completed_at':time.time(),'profile_rows':len(profiles),'niah_rows':len(niah)})
if __name__=='__main__':main()
