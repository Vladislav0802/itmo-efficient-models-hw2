import json
from pathlib import Path

def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def config(path):
    c = json.loads(Path(path).read_text())
    assert c['world_size'] == 2 and c['global_batch'] == 8
    assert c['seq_len'] == 2048
    if c.get('initialization') != 'base':
        assert Path(c['checkpoint']).joinpath('config.json').exists()
    return c
