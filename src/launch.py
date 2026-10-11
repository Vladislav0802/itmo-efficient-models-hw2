import argparse,json,os,sys,time,shutil,subprocess,signal,zipfile
from pathlib import Path
from common import dump

def run(args,root,label):
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',DS_BUILD_OPS='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (root/(label+'.log')).open('w') as log:
        p=subprocess.Popen(args,cwd=Path(__file__).parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            rc=p.wait()
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if rc:raise RuntimeError(f'{label} failed; see {root}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--hours',type=float,default=9);p.add_argument('--eval-minutes',type=float,default=60);p.add_argument('--max-new-steps',type=int,default=10000);p.add_argument('--output-root',default='runs/stage2')
    a=p.parse_args();source=Path(a.input).resolve();c=json.loads((source/'config.json').read_text());meta=json.loads((source/'checkpoint_metadata.json').read_text());tag=meta['full_resume_tag']
    hf=source/'hf'/tag;resume=source/'resume_state'
    assert (hf/'config.json').exists() and (resume/tag/'COMPLETE').exists(),'Need complete HF + optimizer checkpoint'
    assert list((resume/tag).glob('*optim_states.pt')),'Optimizer state missing'
    for rank in range(2):assert (resume/tag/f'data_state_rank{rank}.pt').exists()
    manifest=json.loads((source/'cache/manifest.json').read_text())
    for name in ['dev_doc_hashes','test_doc_hashes','dev_source_shard','test_source_shard']:
        assert name in manifest, 'Missing source manifest field: '+name
    assert int(manifest['dev_source_shard'])==int(manifest['test_source_shard'])+1,'Unexpected source shard layout'
    prior=json.loads((source/'training_result.json').read_text());assert prior['final_step']==meta['step']
    root=Path(a.output_root).resolve()/time.strftime('%Y%m%d_%H%M%S',time.gmtime());root.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(root).free>12*2**30,'Need 12 GiB free working disk; input stays read-only'
    started=time.time();c.update(source_output=str(source),run_dir=str(source),night_dir=str(root),initialization='full_resume',checkpoint=str(hf),hf_checkpoint=str(hf),full_resume_root=str(resume),full_resume_tag=tag,start_step=meta['step'],max_absolute_steps=meta['step']+a.max_new_steps,stop_train_at=started+a.hours*3600-900,world_size=2,seq_len=2048,global_batch=8,micro_batch=1,checkpointing=False,learning_rate=3e-5,checkpoint_every=250,eval_every=1000,benchmark_warmup=5,source_shards=(int(manifest['source_shards']) if 'source_shards' in manifest else int(manifest['dev_source_shard'])+1),source_successful_updates=prior.get('successful_updates_total',prior['successful_updates']),source_skipped_updates=prior.get('skipped_updates_total',prior['skipped_updates']),wall_hours=a.hours,evaluation_minutes=a.eval_minutes,stream_restore='deterministic replay with every cached block verified',duplicate_weight_archive=False)
    code=root/'code';shutil.copytree(Path(__file__).parent,code,ignore=shutil.ignore_patterns('__pycache__'))
    # Use byte-identical author modules from the pinned run, no unpinned updates.
    for name in ['drope.py','attention.py']:
        author=source/'code/custom_models'/name
        assert author.read_bytes()==(code/'custom_models'/name).read_bytes(),'Architecture changed: '+name
    dump(root/'config.json',c)
    (root/'environment_pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('CONTINUATION:',root,'from',meta['step'],'LR=3e-5; no warmup restart',flush=True)
    try:
        run([sys.executable,'-u',str(code/'prepare_continue.py'),'--config',str(root/'config.json')],root,'prepare')
        assert time.time()<c['stop_train_at'],'Preparation used training budget'
        run(['deepspeed','--num_gpus=2',str(code/'train_continue.py'),'--config',str(root/'config.json')],root,'training')
        try:
            run([sys.executable,'-u',str(code/'evaluation/run_check.py'),'--input',str(root),'--minutes',str(a.eval_minutes),'--output-root',str(root/'evaluation')],root,'evaluation')
        except Exception as exc:
            (root/'evaluation_error.txt').write_text(str(exc));print('Evaluation incomplete; saved weights retained',flush=True)
    finally:
        # No duplicate 5.4 GiB TAR. Raw HF+resume folders are saved as Notebook Output.
        with zipfile.ZipFile(root/'continuation_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob('*')):
                if f.is_file() and f.suffix in ['.json','.csv','.log','.txt','.py'] and not any(x in f.relative_to(root).parts for x in ['hf','resume_state','cache']):z.write(f,f.relative_to(root))
            if (root/'cache/manifest.json').exists():z.write(root/'cache/manifest.json','cache/manifest.json')
            for f in sorted((root/'evaluation').glob('*/check_results.zip')) if (root/'evaluation').exists() else []:z.write(f,'evaluation/check_results.zip')
    print('DONE: save whole Notebook Output; small report:',root/'continuation_results.zip',flush=True)
if __name__=='__main__':main()
