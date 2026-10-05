"""Read-only prerequisite check. Never installs dependencies or downloads models."""
import importlib.util
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import ENGINE, MODEL_ROOT, hub_cache, python_path

def main():
    print('Python:', python_path())
    print('Engine:', ENGINE)
    print('Whisper cache:', hub_cache())
    dependencies = ['torch', 'torchaudio', 'librosa', 'numpy', 'scipy', 'soundfile',
                    'transformers', 'huggingface_hub', 'einops', 'munch', 'beartype',
                    'rotary_embedding_torch', 'ml_collections', 'loralib', 'julius',
                    'safetensors', 'tqdm', 'yaml', 'matplotlib', 'dac']
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        print('Missing packages:', ', '.join(missing))
        print('Install the packages listed in the project README with this interpreter.')
        return 1
    required = [ENGINE / 'inference.py', ENGINE / 'configs/YingMusic-SVC.yml',
                ENGINE / 'pupu/models/vocoders/gan/generator/pupuvocoder.py']
    required += [MODEL_ROOT / name for name in (
        'YingMusic-SVC-full.pt', 'campplus_cn_common.bin', 'rmvpe/model.pt',
        'pupu/model.safetensors', 'pc_nsf_hifigan/config.json', 'pc_nsf_hifigan/model.ckpt')]
    ok = True
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            print('Missing file:', path)
            ok = False
    from huggingface_hub import try_to_load_from_cache
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        cached = try_to_load_from_cache('openai/whisper-small', name, cache_dir=str(hub_cache()))
        if not isinstance(cached, str) or not Path(cached).is_file():
            print('Missing Whisper cache:', name)
            ok = False
    import torch
    print('PyTorch:', torch.__version__)
    if not torch.cuda.is_available():
        print('No CUDA GPU is available. Configure CUDA PyTorch and your NVIDIA driver.')
        ok = False
    else:
        try:
            test = torch.randn(32, 32, device='cuda')
            (test @ test).sum().item()
            print('GPU:', torch.cuda.get_device_name(0))
        except Exception as exc:
            print('GPU execution failed:', exc)
            ok = False
    if ok:
        os.chdir(ENGINE)
        sys.path.insert(0, str(ENGINE))
        import inference
        from modules.length_regulator import InterpolateRegulator
        print('Environment and model paths are ready. No downloads were performed.')
    return 0 if ok else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('Environment check failed:', exc)
        sys.exit(1)
