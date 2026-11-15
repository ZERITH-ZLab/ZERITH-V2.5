"""Hosted artifact initialization and precision settings remain compatible."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from zerith_sim.models.zerith import ZerithConfig, ZerithPolicy, ZerithPostTrainModel
from zerith_sim.posttrain.prompts import libero_plus_prompt


@pytest.mark.parametrize('bench', ['libero', 'libero-plus', 'metaworld'])
def test_training_initializes_from_hosted_inference_prefixes_and_native_heads(tmp_path, bench):
    cfg = ZerithConfig.tiny(metaworld_native4=bench == 'metaworld',
                             state_codec_embodiments=('metaworld_sawyer_applied4',) if bench == 'metaworld' else ('libero_franka',))
    policy = ZerithPolicy(cfg)
    weights = {k: v.contiguous().clone() for k, v in policy.state_dict().items()}
    path = tmp_path / 'model.safetensors'
    save_file(weights, str(path))
    model = ZerithPostTrainModel(cfg.as_dict(), pretrained_checkpoint=str(path))
    codec_before = {k: v.clone() for k, v in model.state_codec.state_dict().items()}
    model.load_pretrained_weights()
    for key, value in model.state_dict().items():
        if key.startswith('state_codec.'):
            torch.testing.assert_close(value, codec_before[key.removeprefix('state_codec.')])
        else:
            torch.testing.assert_close(value, weights[key.replace('teacher_carrier.', 'world.')])


@pytest.mark.parametrize('dtype_override,expected', [('auto', torch.float32), ('bfloat16', torch.bfloat16)])
def test_serve_obeys_release_precision_and_resets_both_encoders(tmp_path, monkeypatch, dtype_override, expected):
    import flask
    import zerith_sim.models.zerith as models
    import zerith_sim.vision as vision
    import zerith_sim.vision.wan_assets as wan_assets
    from zerith_sim.posttrain import serve, visual

    (tmp_path / 'manifest.json').write_text(json.dumps({'action_schema': {'benchmark': 'libero'}}))
    (tmp_path / 'inference.json').write_text(json.dumps({'benchmark': 'libero', 'policy_dtype': 'float32'}))
    policy = SimpleNamespace(config=SimpleNamespace(action_flow_steps=10, action_prediction_steps=16,
                                                    action_chunk_size=4), reset_inference=lambda seed: None)
    calls, resets, apps = [], [], []
    def loader(*args, **kwargs):
        calls.append(kwargs)
        return policy, {}
    class Encoder:
        def __init__(self, *args): pass
        def reset(self): resets.append(self)
    class Text:
        def __init__(self): self.model = self
        def to(self, device): return self
        def cpu(self): return self
        def __call__(self, prompts, device): return [torch.zeros(3, 4096)]
    monkeypatch.setattr(models, 'load_deployment_artifact', loader)
    monkeypatch.setattr(vision, 'load_wan22_vae', lambda *args: object())
    monkeypatch.setattr(visual, 'StreamingWanVAEEncoder', Encoder)
    monkeypatch.setattr(visual, 'StreamingVGGTGeometryEncoder', Encoder)
    monkeypatch.setattr(wan_assets, '_load_text_encoder', lambda *args: Text())
    monkeypatch.setattr(flask.Flask, 'run', lambda app, **kwargs: apps.append(app))
    args = SimpleNamespace(artifact=str(tmp_path), dtype=dtype_override, device='cpu',
                           wan_model='wan', wan_repo='source', vggt_model='vggt', vggt_repo='source',
                           host='127.0.0.1', port=8000)
    serve.serve(args)
    assert calls[0]['dtype'] == expected
    assert calls[0]['verify_weight_sha256'] is True
    assert apps[0].test_client().post('/reset', json={'prompt': 'move the block'}).status_code == 200
    assert len(resets) == 2 and resets[0] is not resets[1]


@pytest.mark.parametrize('prompt,expected', [
    ('move the bowl', 'move the bowl'),
    ('move the bowl table -1', 'move the bowl'),
    ('move the bowl light 2', 'move the bowl'),
    ('move the bowl carefully', 'move the bowl carefully'),
    ('move the bowl table -1 extra words', 'move the bowl table -1 extra words'),
    ('move another bowl table -1', 'move another bowl table -1'),
])
def test_plus_prompt_adapter_strips_only_complete_recognized_suffixes(prompt, expected):
    assert libero_plus_prompt(prompt, ['move the bowl'], r'(?:table -?\d+|light -?\d+)') == expected


def test_online_latent_matches_training_cache_precision():
    from zerith_sim.posttrain.visual import policy_latent

    raw = torch.tensor([1.0003, -2.0007], dtype=torch.float32)
    # Prepared training episodes persist VAE outputs as numpy float16 arrays.
    stored = raw.numpy().astype('float16')
    expected = torch.from_numpy(stored).float()
    assert not torch.equal(raw, expected)
    torch.testing.assert_close(policy_latent(raw, torch.float32), expected, rtol=0, atol=0)
    assert policy_latent(raw, torch.bfloat16).dtype == torch.bfloat16
    with pytest.raises(ValueError, match='FP16'):
        policy_latent(torch.tensor([1e6]), torch.float32)
