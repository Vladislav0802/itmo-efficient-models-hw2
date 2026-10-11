import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from prepare_data import Stream
from common import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=json.loads(Path(p.parse_args().config).read_text())
    out=Path(c['night_dir'])/'cache';out.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(c['checkpoint'])
    ds=load_dataset(c['dataset'],revision=c['dataset_revision'],split='train',streaming=True)
    count=(c['max_absolute_steps']-c['start_step'])*4
    assert ds.num_shards>=3 and len(tok)<65536
    dev=Stream(ds.shard(ds.num_shards,ds.num_shards-1),tok,2048)
    dev_blocks=[dev.chunk() for _ in range(c['dev_blocks'])]
    dev_hashes=dev.hashes
    np.save(out/'dev.npy',np.asarray(dev_blocks,dtype=np.uint16))
    print('Dev data ready; preparing training cache.',flush=True)
    hashes=set(); started=time.monotonic()
    for rank in range(2):
        stream=Stream(ds.skip(1000).shard(2,rank),tok,2048)
        array=np.lib.format.open_memmap(out/f'train_rank{rank}.npy',mode='w+',dtype=np.uint16,shape=(count,2048))
        states={};stride=c['checkpoint_every']*4
        for i in range(count):
            array[i]=stream.chunk()
            if (i+1)%stride==0 or i+1==count: states[str(i+1)]=stream.state()
            if (i+1)%2000==0:
                print(f'Data rank{rank}: {i+1}/{count} blocks, {time.monotonic()-started:.0f}s',flush=True)
        array.flush();del array
        hashes.update(stream.hashes)
        dump(out/f'data_state_rank{rank}.json',states)
    if hashes&dev_hashes: raise RuntimeError('Train/dev overlap in continuation')
    # Different physical source shard from dev. Nested prefixes make contexts comparable.
    test=Stream(ds.shard(ds.num_shards,ds.num_shards-2),tok,8192)
    blocks=[test.chunk() for _ in range(c['test_blocks'])]
    if test.hashes&(hashes|dev_hashes): raise RuntimeError('Train/test or dev/test document overlap')
    np.save(out/'test.npy',np.asarray(blocks,dtype=np.uint16))
    dump(out/'manifest.json',{'train_blocks_per_rank':count,'test_blocks':len(blocks),
         'dataset_revision':c['dataset_revision'],'test_source_shard':ds.num_shards-2,
         'test_doc_hashes':sorted(test.hashes),'continuation_overlap':0,
         'dev_doc_hashes':sorted(dev_hashes),'train_doc_hashes_count':len(hashes),
         'dev_source_shard':ds.num_shards-1,'note':'All cached train/dev/test document hashes checked for overlap.'})
    print('NIGHT DATA READY',flush=True)
if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
