"""Run extra diagnostics on 2 GPUs. Parent never initializes CUDA."""
import argparse,json,os,sys,time,subprocess,signal,hashlib,zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
import pandas as pd
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--minutes',type=float,default=45);p.add_argument('--output-root',default='runs/extra');a=p.parse_args()
    begin=time.time();assert 0<a.minutes<=60
    source=Path(a.input).resolve();old=json.loads((source/'config.json').read_text());meta=json.loads((source/'checkpoint_metadata.json').read_text())
    root=Path(a.output_root).resolve()/time.strftime('%Y%m%d_%H%M%S',time.gmtime());root.mkdir(parents=True)
    hf=source/'hf'/meta['full_resume_tag'];test=source/'cache/test.npy'
    assert (hf/'config.json').exists() and test.exists()
    # Do not silently evaluate a different attention implementation.
    for name in ['drope.py','attention.py']:
        assert (source/'code/custom_models'/name).read_bytes()==(Path(__file__).parent/'custom_models'/name).read_bytes()
    base=snapshot_download(old['model_name'],revision=old['model_revision'],allow_patterns=['config.json','generation_config.json','model*.safetensors','tokenizer*','special_tokens_map.json','vocab.json','merges.txt'])
    c=dict(base=base,hf=str(hf),test=str(test),out=str(root),step=meta['step'],deadline=begin+a.minutes*60,test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),protocol='12 short controls, 4 no-needle, 8 main samples per depth; fixed L/2048 factors. YaRN metadata corrected; generation may reach L+24.')
    (root/'config.json').write_text(json.dumps(c,indent=2));procs=[]
    try:
        for gpu,role in enumerate(['rope','drope']):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            log=(root/f'{role}.log').open('w');proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'extra_worker.py'),str(root/'config.json'),role],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append((role,proc,log))
        last={role:0 for role,_,_ in procs}
        while any(p.poll() is None for _,p,_ in procs) and time.time()<c['deadline']+30:
            for role,_,_ in procs:
                lines=(root/f'{role}.log').read_text(errors='replace').splitlines()
                for line in lines[last[role]:]:print(role,line,flush=True)
                last[role]=len(lines)
            time.sleep(5)
    finally:
        for _,proc,log in procs:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        (root/'worker_status.json').write_text(json.dumps({r:p.returncode for r,p,_ in procs},indent=2))
        lm=[];niah=[]
        for role in ['rope','drope']:
            for name,rows in [('lm',lm),('niah',niah)]:
                f=root/f'{role}_{name}.json'
                if f.exists():rows.extend(json.loads(f.read_text()))
        if lm:
            df=pd.DataFrame(lm).groupby(['model','method','length','factor'],as_index=False).agg(ce=('ce','mean'),tail512_ce=('tail512_ce','mean'),blocks=('block','count'));df['ppl']=np.exp(df.ce);df['complete']=df.blocks==8;df.to_csv(root/'lm_summary.csv',index=False)
        if niah:
            df=pd.DataFrame(niah);df=df[df.needle_present].groupby(['model','kind','length','depth'],as_index=False).agg(correct=('correct','sum'),n=('correct','count'));df['accuracy']=df.correct/df.n;df.to_csv(root/'niah_summary.csv',index=False)
        with zipfile.ZipFile(root/'extra_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in root.iterdir():
                if f.is_file() and f.suffix in ['.json','.csv','.log']:z.write(f,f.name)
        print('Saved:',root/'extra_results.zip',flush=True)
if __name__=='__main__':main()
