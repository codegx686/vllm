from modelscope import snapshot_download
from pathlib import Path
import json

root = Path('/root/autodl-tmp/vllm-budget-lab')
path = snapshot_download('Qwen/Qwen3-1.7B', local_dir=str(root / 'models/Qwen3-1.7B'))
(root / 'logs/model-download.json').write_text(json.dumps({'model': 'Qwen/Qwen3-1.7B', 'path': path}))
print('MODEL_READY', path, flush=True)
