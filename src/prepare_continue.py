import argparse,json,os,sys,time,shutil
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from prepare_data import Stream
from common import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    c=json.loads(Path(p.parse_args().config).read_text());source=Path(c['source_output']);out=Path(c['night_dir'])/'cache';out.mkdir()
    tok=AutoTokenizer.from_pretrained(c['hf_checkpoint']);assert len(tok)<65536
    ds=load_dataset(c['dataset'],revision=c['dataset_revision'],split='train',streaming=True)
    assert ds.num_shards==c['source_shards'], 'Dataset shard layout changed'
    consumed=c['start_step']*4;count=(c['max_absolute_steps']-c['start_step'])*4
    manifest=json.loads((source/'cache/manifest.json').read_text())
    excluded=set(manifest['dev_doc_hashes'])|set(manifest['test_doc_hashes'])
    hashes=set();begin=time.monotonic()
    for rank in range(2):
        stream=Stream(ds.skip(1000).shard(2,rank),tok,2048)
        old=np.load(source/'cache'/f'train_rank{rank}.npy',mmap_mode='r')
        assert old.shape[0]>=consumed
        # Replay deterministically: old snapshots used mutable buffer references.
        # Compare every consumed block and cached overlap, then save copied buffers.
        for i in range(consumed):
            block=stream.chunk()
            if not np.array_equal(block,old[i]):raise RuntimeError(f'Stream mismatch rank={rank} block={i}; training has NOT started')
            if (i+1)%10000==0:print(f'REPLAY rank{rank}: {i+1}/{consumed}, {time.monotonic()-begin:.0f}s',flush=True)
        array=np.lib.format.open_memmap(out/f'train_rank{rank}.npy',mode='w+',dtype=np.uint16,shape=(count,2048))
        states={'0':stream.state()};stride=c['checkpoint_every']*4
        for i in range(count):
            block=stream.chunk()
            if consumed+i<len(old) and not np.array_equal(block,old[consumed+i]):raise RuntimeError('Unconsumed cached overlap mismatch')
            array[i]=block
            if (i+1)%stride==0 or i+1==count:states[str(i+1)]=stream.state()
            if (i+1)%5000==0:print(f'PREPARE rank{rank}: {i+1}/{count}, {time.monotonic()-begin:.0f}s',flush=True)
        array.flush();del array,old
        hashes.update(stream.hashes);dump(out/f'data_state_rank{rank}.json',states)
    assert not hashes&excluded,'Train/dev/test text overlap detected'
    for name in ['dev.npy','test.npy']:shutil.copy2(source/'cache'/name,out/name)
    dump(out/'manifest.json',dict(manifest,train_blocks_per_rank=count,start_absolute_step=c['start_step'],stream_replayed_and_verified=True,train_doc_hashes_count=len(hashes),continuation_overlap=0))
    print('DATA READY: exact consumed stream and remaining cache verified',flush=True)
if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
