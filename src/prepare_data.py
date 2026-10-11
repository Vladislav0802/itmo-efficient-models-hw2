import argparse, hashlib, json, time, os, sys
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from common import config, dump

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

class Stream:
    def __init__(self, ds, tokenizer, length):
        self.it, self.tok, self.length = iter(ds), tokenizer, length
        self.docs_seen, self.buffer = 0, []
        self.hashes = set()
    def chunk(self):
        while len(self.buffer) < self.length:
            row = next(self.it)
            self.docs_seen += 1
            text = row.get('text')
            if not text or len(text.strip()) < 100:
                continue
            self.hashes.add(digest(text))
            self.buffer.extend(self.tok(text, add_special_tokens=False)['input_ids'])
            if self.tok.eos_token_id is not None:
                self.buffer.append(self.tok.eos_token_id)
        out = self.buffer[:self.length]
        del self.buffer[:self.length]
        return out
    def state(self):
        return {'docs_seen': self.docs_seen, 'buffer': list(self.buffer)}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    config_path = Path(p.parse_args().config)
    c = config(config_path)
    root = Path(c['run_dir']); cache = root / 'cache'; cache.mkdir(parents=True, exist_ok=True)
    if c.get('initialization') == 'base':
        snapshot = snapshot_download(c['model_name'], allow_patterns=[
            'config.json', 'generation_config.json', 'model*.safetensors',
            'tokenizer*', 'special_tokens_map.json', 'vocab.json', 'merges.txt'])
        c['checkpoint'] = snapshot
        c['model_revision'] = Path(snapshot).name
        c['dataset_revision'] = HfApi().dataset_info(c['dataset']).sha
        dump(config_path, c)
        print('Base model cached:', snapshot, flush=True)
    tok = AutoTokenizer.from_pretrained(c['checkpoint'])
    assert len(tok) < 65536, 'uint16 cache requires vocab < 65536'
    ds = load_dataset(c['dataset'], revision=c.get('dataset_revision'), split='train', streaming=True)
    assert ds.num_shards >= 3, 'Need a separate source shard for dev'
    replay = c['pilot_step'] * c['global_batch'] // c['world_size']
    count = c['lr_steps'] * c['global_batch'] // c['world_size']
    train_hashes = set()
    for rank in range(c['world_size']):
        # Reproduce the pilot's exact ordering with the same datasets version.
        stream = Stream(ds.skip(1000).shard(2, rank), tok, c['seq_len'])
        print(f'rank {rank}: replay {replay} cached-pilot chunks', flush=True)
        for i in range(replay):
            stream.chunk()
            if (i + 1) % 1000 == 0: print(f'  replay {i+1}/{replay}', flush=True)
        blocks, states = [], {}
        stride = c['checkpoint_every'] * c['global_batch'] // c['world_size']
        for i in range(count):
            blocks.append(stream.chunk())
            if (i+1) % stride == 0 or i+1 == count:
                states[str(i+1)] = stream.state()
        np.save(cache / f'train_rank{rank}.npy', np.asarray(blocks, dtype=np.uint16))
        dump(cache / f'data_state_rank{rank}.json', states)
        train_hashes.update(stream.hashes)
    # Last physical source shard: disjoint from the beginning of each pilot rank.
    dev_stream = Stream(ds.shard(ds.num_shards, ds.num_shards - 1), tok, c['seq_len'])
    dev = []
    while len(dev) < c['dev_blocks']:
        block = dev_stream.chunk()
        if dev_stream.hashes & train_hashes:
            raise RuntimeError('Train/dev document overlap detected; choose another dev source shard')
        dev.append(block)
    np.save(cache / 'dev.npy', np.asarray(dev, dtype=np.uint16))
    manifest = {'dataset': c['dataset'], 'dataset_revision': c.get('dataset_revision'),
                'source_shards': ds.num_shards,
                'dev_source_shard': ds.num_shards - 1,
                'train_blocks_per_rank': count, 'dev_blocks': len(dev),
                'pilot_replayed_chunks_per_rank': replay,
                'train_doc_hashes_count': len(train_hashes),
                'dev_doc_hashes': sorted(dev_stream.hashes),
                'overlap': 0}
    dump(cache / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
    # All cache/manifest files have been closed by this point. Streaming readers
    # can otherwise block interpreter shutdown in their background cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
