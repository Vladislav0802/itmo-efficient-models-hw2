import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed
from transformers import AutoTokenizer
from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM
from common import config, dump

def ce(logits, labels):
    # Only the current token slice is upcast; no full [B,T,V] FP32 tensor.
    values = []
    for start in range(0, labels.shape[1], 128):
        a = logits[:, start:start+128].float()
        b = labels[:, start:start+128]
        values.append(F.cross_entropy(a.reshape(-1, a.shape[-1]), b.reshape(-1), reduction='sum'))
    return torch.stack(values).sum() / labels.numel()

@torch.inference_mode()
def evaluate(model, data, device):
    model.eval(); total = 0.0
    for block in data:
        ids = torch.tensor(np.array(block, dtype=np.int64), device=device).unsqueeze(0)
        logits = model(input_ids=ids, use_cache=False).logits
        total += ce(logits[:, :-1], ids[:, 1:]).item()
    return total / len(data)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--local_rank', type=int, default=0)
    args = p.parse_args(); c = config(args.config)
    outdir=Path(c['night_dir']); t={'micro_batch':c['micro_batch'],'checkpointing':c['checkpointing'],'lr':c['learning_rate'],'kind':'night'}
    deepspeed.init_distributed(); rank = dist.get_rank()
    assert dist.get_world_size() == 2
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])); torch.cuda.set_device(device)
    random.seed(42 + rank); np.random.seed(42 + rank)
    torch.manual_seed(42 + rank); torch.cuda.manual_seed_all(42 + rank)
    batch = t['micro_batch']; accum = c['global_batch'] // (2 * batch)
    cfg=DroPELlamaConfig.from_pretrained(c['checkpoint']);cfg.attention_type='qk_norm_nope'
    model = DroPELlamaForCausalLM.from_pretrained(c['checkpoint'],config=cfg,dtype=torch.float32,attn_implementation='sdpa')
    assert model.config.attention_type == 'qk_norm_nope'
    assert all(type(x.self_attn).__name__ == 'QKNormNoPELlamaAttention' for x in model.model.layers)
    if t['checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0, foreach=False)
    ds_cfg = {'train_micro_batch_size_per_gpu': batch, 'gradient_accumulation_steps': accum,
              'train_batch_size': c['global_batch'], 'gradient_clipping': 1.0,
              'steps_per_print': 10000000, 'zero_allow_untested_optimizer': True,
              'fp16': {'enabled': True, 'loss_scale': 0, 'initial_scale_power': 12},
              'zero_optimization': {'stage': 1, 'overlap_comm': True,
                                    'reduce_scatter': True, 'contiguous_gradients': True}}
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_cfg)
    engine.train(); assert engine.module.training
    data = np.load(outdir / 'cache' / f'train_rank{rank}.npy', mmap_mode='r')
    dev = np.load(outdir / 'cache' / 'dev.npy', mmap_mode='r')
    initial_dev_loss=None
    dist.barrier()
    if rank==0:
        initial_dev_loss=evaluate(engine.module,dev,device)
        print(f'Initial QKNoPE dev CE={initial_dev_loss:.4f}',flush=True)
    dist.barrier();engine.train()
    curve=[] if rank else [{'step':0,'dev_loss':initial_dev_loss,'dev_ppl':float(np.exp(initial_dev_loss))}]
    cursor = 0; rows = []; durations = []; skips_start = engine.skipped_steps
    def save_checkpoint(step):
        resume = outdir/'resume_state'; tag = f"step_{step:05d}"
        engine.save_checkpoint(str(resume), tag=tag, save_latest=False,
            client_state={'absolute_step': step,
                          'tuning_config': t, 'source_config': c,
                          'tokens_seen_global': (step)*c['global_batch']*c['seq_len'],
                          'optimizer_restarted_from_pilot': False})
        states = json.loads((outdir/'cache'/f'data_state_rank{rank}.json').read_text())
        # Restore a timed stop exactly: prepend cached unconsumed tokens to
        # the next saved stream state, whose docs_seen includes those documents.
        boundary=min(int(k) for k in states if int(k)>=cursor)
        state=states[str(boundary)]
        state['buffer']=np.array(data[cursor:boundary],dtype=np.int64).reshape(-1).tolist()+state['buffer']
        state.update(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(state, resume/tag/f'data_state_rank{rank}.pt')
        dist.barrier()
        if rank == 0:
            # Metadata points at a new checkpoint only after weights also finish.
            (resume/'latest_tag.txt').write_text(tag)
            hf = outdir/'hf'/tag
            engine.module.config.use_cache = True
            engine.module.save_pretrained(hf, safe_serialization=True)
            engine.module.config.use_cache = False
            AutoTokenizer.from_pretrained(c['checkpoint']).save_pretrained(hf)
            assert list(hf.glob('*.safetensors')) and (hf/'config.json').exists()
            (resume/tag/'COMPLETE').write_text(tag)
            dump(outdir/'checkpoint_metadata.json', {'step': step,
                  'hf_checkpoint': str(hf), 'full_resume_root': str(resume),
                  'full_resume_tag': tag, 'cache_cursor_per_rank': cursor,
                  'initialization': c.get('initialization'), 'trial': t})
            import shutil
            # Keep one complete checkpoint; delete old only after new weights/state finish.
            for folder in [resume,outdir/'hf']:
                for old in sorted(folder.glob('step_*'))[:-1]: shutil.rmtree(old)
            print('CHECKPOINT SAVED:', tag, str(hf), flush=True)
        dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    final_step=c['start_step']
    for step in range(c['start_step']+1,c['max_absolute_steps']+1):
        lr=t['lr']*min(1.0,step/c['warmup_steps'])
        for group in engine.optimizer.param_groups:group['lr']=lr
        # Rank 0 alone decides the wall-clock stop, then all ranks obey.
        stop=torch.tensor(int(time.time()>=c['stop_train_at']) if rank==0 else 0,device=device)
        dist.broadcast(stop,src=0)
        if stop.item(): break
        torch.cuda.synchronize(device); begin = time.perf_counter(); losses = []
        for _ in range(accum):
            array = np.array(data[cursor:cursor+batch], dtype=np.int64)
            assert len(array) == batch, 'Cache exhausted'
            cursor += batch
            ids = torch.tensor(array, device=device)
            logits = engine(input_ids=ids, use_cache=False).logits
            loss = ce(logits[:, :-1], ids[:, 1:])
            finite = torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item(): raise RuntimeError('Non-finite loss on at least one rank')
            losses.append(loss.detach())
            engine.backward(loss); engine.step()
            del logits, loss
        avg = torch.stack(losses).mean(); dist.all_reduce(avg); avg /= 2
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter()-begin, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        sec = elapsed.item()
        if step > c['benchmark_warmup']: durations.append(sec)
        if rank == 0:
            rows.append({'step': step, 'loss': avg.item(), 'lr': lr, 'seconds': sec})
            if step == 1 or step % 20 == 0:
                print(f"night {step}/{c['max_absolute_steps']} loss={avg.item():.4f} {sec:.2f}s", flush=True)
                dump(outdir / 'training_log.json', rows)
        final_step=step
        if step%c['checkpoint_every']==0: save_checkpoint(step)
        if step%c['eval_every']==0:
            dist.barrier()
            if rank==0:
                val=evaluate(engine.module,dev,device)
                curve.append({'step':step,'dev_loss':val,'dev_ppl':float(np.exp(val))})
                dump(outdir/'dev_curve.json',curve)
                print(f'DEV step={step} CE={val:.4f}',flush=True)
            dist.barrier();engine.train()
    if final_step==c['start_step']: raise RuntimeError('No time for continuation')
    # Cursor at a non-periodic timed stop needs its exact streaming state too.
    if final_step%c['checkpoint_every']!=0: save_checkpoint(final_step)
    skipped = torch.tensor(engine.skipped_steps-skips_start, device=device)
    dist.all_reduce(skipped, op=dist.ReduceOp.MAX)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device)/2**30, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank==0:
        mean=sum(durations)/len(durations) if durations else None
        result={'status':'OK','start_step':c['start_step'],'final_step':final_step,
                'additional_steps':final_step-c['start_step'],'skipped_updates':skipped.item(),
                'successful_updates':final_step-c['start_step']-skipped.item(),
                'tokens_seen_global':final_step*8*2048,'mean_seconds_per_step':mean,
                'peak_allocated_gb':peak.item(),'learning_rate':t['lr']}
        dump(outdir/'training_log.json',rows);dump(outdir/'training_result.json',result)
        print(json.dumps(result,indent=2),flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__': main()
