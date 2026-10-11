import argparse,json,sys,os,time,subprocess,hashlib,shutil,signal,zipfile,csv,math
from pathlib import Path
from huggingface_hub import snapshot_download
from common import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--minutes',type=float,default=55);p.add_argument('--output-root',default='runs/check')
    a=p.parse_args();begin=time.time();source=Path(a.input).resolve()
    old=json.loads((source/'config.json').read_text());meta=json.loads((source/'checkpoint_metadata.json').read_text())
    root=Path(a.output_root).resolve()/time.strftime('%Y%m%d_%H%M%S',time.gmtime());root.mkdir(parents=True)
    for name in ['test.npy','dev.npy']:assert (source/'cache'/name).exists(), 'Connect the whole night Notebook Output, including cache/'+name
    hf=source/'hf'/meta['full_resume_tag'];assert (hf/'config.json').exists()
    # Pin original architecture modules to the exact training code.
    for name in ['drope.py','attention.py']:
        archived=source/'code/custom_models'/name;bundled=Path(__file__).parent/'custom_models'/name
        assert archived.exists() and archived.read_bytes()==bundled.read_bytes(),'Training attention source differs: '+name
    base=snapshot_download(old['model_name'],revision=old['model_revision'],allow_patterns=['config.json','generation_config.json','model*.safetensors','tokenizer*','special_tokens_map.json','vocab.json','merges.txt'])
    c={'source_output':str(source),'out_dir':str(root),'step':meta['step'],'hf_path':str(hf),
       'base_path':base,'test_path':str(source/'cache/test.npy'),'dev_path':str(source/'cache/dev.npy'),
       'test_sha256':hashlib.sha256((source/'cache/test.npy').read_bytes()).hexdigest(),
       'test_blocks':8,'niah_samples':2,'worker_deadline':begin+a.minutes*60,
       'dataset_revision':old['dataset_revision'],'repo_commit':old['repo_commit'],
       'paper':'https://arxiv.org/html/2512.12167v1#A3.SS2',
       'protocol_note':'NIAH uses FineWeb test haystack, not paper Paul Graham corpus. Small diagnostic sample.'}
    dump(root/'check_config.json',c);print('CHECK OUTPUT:',root,flush=True)
    code=root/'code';shutil.copytree(Path(__file__).parent,code,ignore=shutil.ignore_patterns('__pycache__'))
    procs=[];logs=[]
    for index,name in enumerate(['rope','drope']):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(index),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        log=(root/(name+'.log')).open('w');logs.append(log)
        proc=subprocess.Popen([sys.executable,'-u',str(code/'worker.py'),'--config',str(root/'check_config.json'),'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append((name,proc))
    statuses={};last={name:0 for name,_ in procs}
    try:
        while any(proc.poll() is None for _,proc in procs):
            if time.time()>c['worker_deadline']+60:
                for name,proc in procs:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        statuses[name]='TIMEOUT; partial JSON retained'
                break
            for name,_ in procs:
                lines=(root/(name+'.log')).read_text(errors='replace').splitlines()
                new=lines[last[name]:];last[name]=len(lines)
                selected=[l for l in new if any(w in l for w in ['INTEGRITY','PROFILE','NIAH','BETA','Error','Traceback'])]
                for line in selected[-8:]:print(line,flush=True)
            time.sleep(5)
    except BaseException:
        for _,proc in procs:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        raise
    finally:
        for log in logs:log.close()
    for name,proc in procs:statuses.setdefault(name,proc.returncode)
    dump(root/'worker_status.json',statuses)
    aggregate(root)
    print('CHECK DONE:',root/'check_results.zip','statuses:',statuses,flush=True)

def aggregate(root):
    summary=[];niah=[];segments=[]
    for name in ['rope','drope']:
        folder=root/name
        if (folder/'profile.json').exists():
            rows=json.loads((folder/'profile.json').read_text());groups={}
            for row in rows:
                key=(row['model'],row['length'],row['variant'],row['beta']);groups.setdefault(key,[]).append(row)
                for segment in row['segments']:segments.append({k:row[k] for k in ['model','length','variant','beta','block']}|segment)
            for (model,length,variant,beta),values in groups.items():
                ce=sum(r['ce'] for r in values)/len(values)
                summary.append({'model':model,'length':length,'variant':variant,'beta':beta,'blocks':len(values),'ce':ce,'ppl':math.exp(ce),'tail512_ce':sum(r['tail512_ce'] for r in values)/len(values)})
        if (folder/'niah.json').exists():niah+=json.loads((folder/'niah.json').read_text())
    grouped={}
    for row in niah:
        key=(row['model'],row['kind'],row['length'],row['depth'],row['variant']);grouped.setdefault(key,[]).append(row)
    retrieval=[{'model':model,'kind':kind,'length':length,'depth':depth,'variant':variant,'n':len(values),'correct':sum(r['correct'] for r in values),'accuracy':sum(r['correct'] for r in values)/len(values)}
               for (model,kind,length,depth,variant),values in grouped.items()]
    for name,rows in [('lm_summary',summary),('position_losses',segments),('niah_summary',retrieval)]:
        dump(root/(name+'.json'),rows)
        if rows:
            with (root/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Reproducibility check against the already saved evaluation; only unscaled/paper rows.
    previous=root/'prior_comparison.json';comparison=[]
    cfg=json.loads((root/'check_config.json').read_text())
    oldfile=Path(cfg['source_output'])/'test_results.json'
    if oldfile.exists():
        old=json.loads(oldfile.read_text())
        for row in summary:
            if row['variant'] not in ['unscaled','paper']:continue
            model='RoPE_base' if row['model']=='RoPE_base' else 'DroPE_final'
            matches=[r for r in old if r['model']==model and r['length']==row['length'] and r['attention_scaled']==(row['variant']=='paper')]
            if matches:comparison.append({'model':row['model'],'length':row['length'],'variant':row['variant'],'new_ce':row['ce'],'previous_ce':matches[0]['ce'],'abs_delta':abs(row['ce']-matches[0]['ce']),'blocks':row['blocks']})
    dump(previous,comparison)
    with zipfile.ZipFile(root/'check_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ['.json','.csv','.log','.py']:z.write(path,path.relative_to(root))

if __name__=='__main__':main()
