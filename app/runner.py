"""Adapt the author pipeline to structured progress without modifying its models."""
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import ENGINE

def emit(stage, percent, message, indeterminate=False, **extra):
    print('@@PUPU@@' + json.dumps(dict(stage=stage, percent=round(percent, 1), message=message,
                                     indeterminate=indeterminate, **extra), ensure_ascii=False), flush=True)

def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    engine = ENGINE
    os.chdir(engine)
    sys.path.insert(0, str(engine))
    emit('loading', 0, '正在加载运行环境与模型，首次启动需要稍等…', True)
    import inference
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用的 NVIDIA 显卡，请检查显卡驱动。')
    args = inference.finalize_inference_args(inference.build_parser().parse_args([
        '--source', config['source'], '--target', config['target'], '--output', config['output'],
        '--pitch-shift', str(config['pitch']), '--diffusion-steps', str(config['steps']),
        '--inference-cfg-rate', str(config['cfg']), '--vocoder-dtype', config['precision'],
    ]))
    args.uuid = config['id'][:8]
    inference.preflight_check(args)
    bundle = inference.load_models_api(args, device=args.cuda)
    emit('prepare', 10, '模型已就绪，正在分析人声、音高和参考音色…', True)
    original_prepare = inference.prepare_inference_context
    original_chunk = inference.infer_mel_chunk
    cfm = bundle['model'].cfm
    original_cfm = cfm.inference
    progress = {}

    def prepare(*a, **kw):
        ctx = original_prepare(*a, **kw)
        progress['total'] = ctx.cond.size(1)
        emit('convert', 20, '开始转换音色…')
        return ctx

    def chunk(ctx, arguments, device, start, end):
        progress.update(start=start, end=end)
        return original_chunk(ctx, arguments, device, start, end)

    def step_iterator(iterable):
        values = list(iterable)
        for index, step in enumerate(values):
            yield step
            torch.cuda.synchronize()
            completed = progress['start'] + (progress['end'] - progress['start']) * (index + 1)/len(values)
            emit('convert', 20 + 55*completed/progress['total'],
                 f'正在转换音色 · 当前片段 {index+1}/{len(values)} 步',
                 frames_done=round(completed), frames_total=progress['total'])

    def cfm_inference(*a, **kw):
        kw['pbar'] = step_iterator
        return original_cfm(*a, **kw)

    inference.prepare_inference_context = prepare
    inference.infer_mel_chunk = chunk
    cfm.inference = cfm_inference
    first = bundle['pupu_vocoder'].mel_to_wav
    second = bundle['pcnsf_vocoder'].mel_to_wav

    def pupu(*a, **kw):
        emit('pupu', 76, 'Pupu 正在合成歌声…', True)
        result = first(*a, **kw)
        emit('finish', 86, '正在提取音高并精修声音…', True)
        return result

    def pcnsf(*a, **kw):
        emit('finish', 90, 'PC-NSF 正在精修与导出…', True)
        result = second(*a, **kw)
        emit('finish', 98, '正在保存音频…', True)
        return result

    bundle['pupu_vocoder'].mel_to_wav = pupu
    bundle['pcnsf_vocoder'].mel_to_wav = pcnsf
    output = inference.run_inference(args, bundle, device=args.cuda)
    # Uploaded source files use opaque IDs; restore a readable result filename.
    name = Path(config['source_name'].replace('\\', '/')).stem
    dest = Path(output).parent / f'{name}_pupu_{config["pitch"]:+g}key.flac'
    Path(output).rename(dest)
    emit('done', 100, '转换完成', output=str(dest))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
