import argparse,json,os,sys,subprocess,time,shutil,hashlib
from pathlib import Path
from common import dump

def run(args,root,timeout=None):
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',DS_BUILD_OPS='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (root/(Path(args[0]).stem+'.log')).open('a') as log:
        p=subprocess.Popen(args,cwd=Path(__file__).parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in p.stdout: print(line,end='',flush=True);log.write(line);log.flush()
            rc=p.wait(timeout=timeout)
        except BaseException:
            import signal
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if rc: raise RuntimeError(f'Failed {args}; saved checkpoints remain in {root}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--hours',type=float,default=9);p.add_argument('--max-steps',type=int,default=10000);p.add_argument('--output-root',default='runs/stage1')
    a=p.parse_args();started=time.time()
    c={'initialization':'base','model_name':'HuggingFaceTB/SmolLM-360M',
       'model_revision':'59f7ef243ee09a72cbc14cb054393a3e3b771d41',
       'dataset':'PrimeIntellect/fineweb-edu',
       'dataset_revision':'14efaa24d7dff8a745bf4918e415878546542346',
       'repo_commit':'260a8bb9f4fda0673266ddfee7668915aade8b33',
       'world_size':2,'seq_len':2048,'global_batch':8,'micro_batch':1,
       'checkpointing':False,'learning_rate':a.lr,'start_step':0,'pilot_step':0,
       'warmup_steps':30,'benchmark_warmup':5,'dev_blocks':16}
    assert 0<a.hours<=11 and a.max_steps>0 and a.lr>0
    print('FRESH START: SmolLM-360M -> QKNoPE; LR',a.lr,flush=True)
    from huggingface_hub import snapshot_download
    c['checkpoint']=snapshot_download(c['model_name'],revision=c['model_revision'],allow_patterns=['config.json','generation_config.json','model*.safetensors','tokenizer*','special_tokens_map.json','vocab.json','merges.txt'])
    assert c['micro_batch']==1 and not c['checkpointing'],'Unexpected speed winner; review before night run'
    root=Path(a.output_root).resolve()/time.strftime('%Y%m%d_%H%M%S',time.gmtime());root.mkdir(parents=True,exist_ok=False)
    c.update(night_dir=str(root),max_absolute_steps=a.max_steps,checkpoint_every=250,eval_every=1000,
             stop_train_at=started+a.hours*3600-1800,test_blocks=8,wall_hours=a.hours,
             grad_accum=4,precision='fp16',zero_stage=1,attention_backend='sdpa',
             optimizer={'name':'AdamW','betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0,'gradient_clipping':1.0},
             lr_schedule='30_step_warmup_then_constant')
    assert c['max_absolute_steps']>c['start_step']
    # Space includes one retained checkpoint; old + new coexist only during successful save.
    free=shutil.disk_usage(root).free/2**30
    print(f'Disk free after model download: {free:.2f} GiB',flush=True)
    if free<12: raise RuntimeError(f'Need 12 GiB free; available {free:.2f} GiB. Use a new notebook without old outputs in working.')
    code=root/'code';shutil.copytree(Path(__file__).parent,code,ignore=shutil.ignore_patterns('__pycache__'))
    # Pull only the official modules needed here, at the SAME commit as tuning.
    repo=root/'upstream'
    subprocess.run(['git','init',str(repo)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(repo),'remote','add','origin','https://github.com/SakanaAI/DroPE.git'],check=True)
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',c['repo_commit']],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True,stdout=subprocess.DEVNULL)
    for name in ['drope.py','attention.py']:
        official=repo/'custom_models'/name;bundled=code/'custom_models'/name
        if official.read_bytes()!=bundled.read_bytes(): raise RuntimeError('Pinned upstream module differs from bundled source: '+name)
        shutil.copy2(official,bundled)
    shutil.copy2(repo/'README.md',root/'upstream_README.md')
    subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,stdout=(root/'upstream_commit.txt').open('w'))
    shutil.rmtree(repo)
    dump(root/'config.json',c)
    (root/'environment_pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('NIGHT RUN:',root,flush=True)
    run([sys.executable,'-u',str(code/'prepare_stage1.py'),'--config',str(root/'config.json')],root)
    if time.time()>=c['stop_train_at']: raise RuntimeError('Preparation used training budget; inspect preparation log.')
    run(['deepspeed','--num_gpus=2',str(code/'train_stage1.py'),'--config',str(root/'config.json')],root)
    # Evaluation separate from distributed training; GPU1 stays idle by design here.
    try:
        run([sys.executable,'-u',str(code/'evaluate_stage1.py'),'--config',str(root/'config.json')],root)
    except RuntimeError as exc:
        (root/'evaluation_error.txt').write_text(str(exc))
        print('Evaluation incomplete; packaging trained checkpoint and partial results.',flush=True)
    finally:
        import zipfile
        with zipfile.ZipFile(root/'stage1_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in root.rglob('*'):
                if f.is_file() and f.suffix in ['.json','.csv','.log','.txt','.py'] and not any(x in f.relative_to(root).parts for x in ['hf','resume_state','cache']):z.write(f,f.relative_to(root))
            z.write(root/'cache/manifest.json','cache/manifest.json')
    print('DONE. Save whole output for HF + optimizer; metrics: stage1_results.zip.',flush=True)
if __name__=='__main__':main()
