"""Single-episode HTTP inference with causal video state."""
import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from .recipes import BENCHMARKS
from .sources import libero_state


def serve(args):
    from flask import Flask, request, jsonify
    from zerith_sim.models.zerith import load_deployment_artifact
    from zerith_sim.vision import CameraProfile, load_wan22_vae
    from zerith_sim.vision.wan_assets import _load_text_encoder
    from .visual import StreamingWanVAEEncoder, StreamingVGGTGeometryEncoder, policy_latent
    # Reject incompatible benchmark/H1 artifacts before loading the model or encoders.
    import json
    manifest = json.loads((Path(args.artifact) / 'manifest.json').read_text())
    bench = manifest.get('action_schema', {}).get('benchmark')
    if bench not in BENCHMARKS:
        raise ValueError(f'Unsupported simulation artifact benchmark: {bench!r}; use an H1 bundle with zerith serve for H1')
    spec = BENCHMARKS[bench]
    settings_path = Path(args.artifact) / 'inference.json'
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    if settings.get('benchmark', bench) != bench:
        raise ValueError('inference.json benchmark differs from manifest.json')
    adapter = settings.get('prompt_adapter', {})
    bases = []
    if adapter:
        if bench != 'libero-plus' or adapter.get('mode') != 'liberoplus_suffix_v1':
            raise ValueError('Unsupported release prompt adapter')
        base_path = getattr(args, 'base_instructions', None)
        if not base_path:
            raise ValueError('This LIBERO-Plus release requires --base-instructions with the 40 standard LIBERO task instructions')
        bases = json.loads(Path(base_path).read_text())
        if (not isinstance(bases, list) or len(bases) != 40
                or any(not isinstance(x, str) or not x.strip() for x in bases)
                or len(set(bases)) != len(bases)):
            raise ValueError('Expected 40 unique, nonempty standard LIBERO instructions')
        if not isinstance(adapter.get('suffix_regex'), str):
            raise ValueError('LIBERO-Plus prompt adapter requires suffix_regex')
    dtype_name = getattr(args, 'dtype', 'auto')
    if dtype_name == 'auto':
        dtype_name = settings.get('policy_dtype', 'float32')
    if dtype_name not in ('float32', 'bfloat16'):
        raise ValueError(f'Unsupported policy dtype: {dtype_name}')
    device = torch.device(args.device)
    text_device = torch.device(getattr(args, 'text_device', None) or args.device)
    policy, manifest = load_deployment_artifact(args.artifact, device=device,
                                               dtype=getattr(torch, dtype_name), verify_weight_sha256=True)
    for name, actual in (('action_flow_steps', policy.config.action_flow_steps),
                         ('prediction_steps', policy.config.action_prediction_steps),
                         ('execute_steps', policy.config.action_chunk_size),
                         ('primitive_dt', spec['dt']), ('video_dt', spec['dt'] * 4)):
        if settings.get(name, actual) != actual:
            raise ValueError(f'inference.json {name} differs from the model/benchmark contract')
    profile = CameraProfile(bench, spec['cameras'], tuple(range(len(spec['cameras']))), ('front','left','right'))
    vae = load_wan22_vae(Path(args.wan_model)/'Wan2.2_VAE.pth', args.wan_repo, device)
    visual = StreamingWanVAEEncoder(vae, profile, device)
    geometry = StreamingVGGTGeometryEncoder(Path(args.vggt_repo), Path(args.vggt_model), profile, device)
    text_encoder = _load_text_encoder(Path(args.wan_model), Path(args.wan_repo), torch.device('cpu'))
    @lru_cache(maxsize=256)
    def encode_text(prompt):
        if bases:
            from .prompts import libero_plus_prompt
            prompt = libero_plus_prompt(prompt, bases, adapter['suffix_regex'])
        text_encoder.model.to(text_device)
        with torch.inference_mode():
            value = text_encoder([prompt], text_device)[0].to(device)
        text_encoder.model.cpu()
        if len(value) > 64:
            raise ValueError('Instruction exceeds the 64-token model context')
        context = torch.zeros((1,64,4096), device=device, dtype=torch.float16)
        context[0,:len(value)] = value
        return context
    current = {}
    app = Flask(__name__)
    @app.get('/health')
    def health():
        return jsonify(benchmark=bench)
    @app.post('/reset')
    def reset():
        value = request.get_json()
        visual.reset(); geometry.reset(); policy.reset_inference(int(value.get('seed',0)))
        current['text'] = encode_text(value['prompt'])
        return jsonify(ok=True)
    @app.post('/infer')
    @torch.inference_mode()
    def infer():
        value = request.get_json()
        rgb = np.frombuffer(base64.b64decode(value['images']), np.uint8).reshape(value['shape'])
        state = np.asarray(value['state'], np.float32)
        if bench in ('libero','libero-plus'):
            state = libero_state(state[None])[0]; valid = np.arange(20)<10
        elif bench == 'metaworld':
            from metaworld_benchmark.contract import state20, stored_image
            state = state20(state[:4]); valid = np.isin(np.arange(20), [0,1,2,9])
            rgb = np.stack([stored_image(frame[0]) for frame in rgb])[:,None]
        latent, spatial = visual.encode(rgb, spec['cameras'])
        latent = policy_latent(latent, next(policy.parameters()).dtype)
        geo, gv = geometry.encode(rgb, spec['cameras'])
        pred, _ = policy.act_state_trajectory(latent, spatial, current['text'],
            torch.tensor([spec['dt']],device=device), torch.tensor([spec['dt']*4],device=device),
            torch.from_numpy(state[None]).to(device), torch.from_numpy(valid[None]).to(device),
            torch.zeros(1,dtype=torch.long,device=device), geometry_features=geo,geometry_valid=gv)
        actions = pred[0,:4].float().cpu().numpy()
        if bench in ('libero','libero-plus'): actions = np.clip(actions[:,:7],-1,1)
        elif bench == 'metaworld': actions = np.clip(actions[:,:4],-1,1)
        return jsonify(actions=actions.tolist())
    app.run(host=args.host,port=args.port,threaded=False)
