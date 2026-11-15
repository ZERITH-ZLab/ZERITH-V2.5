"""Read existing caches unchanged while writing current release identifiers."""

import hashlib
import json

import numpy as np
import pytest
import torch

from zerith_sim.vision import wan_assets
from zerith_sim.vision.letterbox import raw_is_preletterboxed


def test_current_cache_environment_takes_precedence_and_old_setting_is_read(monkeypatch, tmp_path):
    monkeypatch.setenv('V52_NODE_ASSET_ROOT', str(tmp_path / 'old'))
    monkeypatch.delenv('ZERITH_NODE_ASSET_ROOT', raising=False)
    assert wan_assets._configured_node_asset_root() == tmp_path / 'old'
    monkeypatch.setenv('ZERITH_NODE_ASSET_ROOT', str(tmp_path / 'current'))
    assert wan_assets._configured_node_asset_root() == tmp_path / 'current'
    monkeypatch.setenv('ZERITH_NODE_ASSET_ROOT', '/')
    with pytest.raises(ValueError, match='specific path'):
        wan_assets._configured_node_asset_root()


@pytest.mark.parametrize('format_name,operator', [
    ('zerith-raw', 'mcap_to_zerith'),
    ('fam-wam-raw', 'mcap_to_fam_wam'),
])
def test_raw_provenance_accepts_both_names_without_changing_padding_checks(format_name, operator):
    conversion = {'operator': operator, 'params': {'resize_mode': 'pad', 'resize_width': 512, 'resize_height': 512}}
    metadata = {'format': format_name, 'format_version': '1.0',
                'camera_specs': {'head': {'width': 640, 'height': 480}},
                'algo_operators_provenance': {'operator_chain': [conversion]}}
    before = json.dumps(metadata, sort_keys=True)
    assert raw_is_preletterboxed(metadata)
    assert json.dumps(metadata, sort_keys=True) == before
    metadata['algo_operators_provenance']['operator_chain'].append(conversion)
    with pytest.raises(ValueError, match='exactly one'):
        raw_is_preletterboxed(metadata)


def test_existing_text_cache_keeps_its_metadata_and_tensor_bytes(monkeypatch, tmp_path):
    tasks = ['pick the block']
    (tmp_path / 'tasks.json').write_text(json.dumps(tasks))
    values = np.ones((1, 4, 4096), dtype=np.float16)
    array = tmp_path / 'text_embeddings.npy'
    np.save(array, values)
    header = {'format': 'fam-wam-umt5-text-embeddings-v1',
              'tasks_sha256': hashlib.sha256(json.dumps(tasks, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
              'asset_receipt_sha256': 'a' * 64, 'dtype': 'float16', 'width': 4096,
              'sequence_length': 4, 'bytes': array.stat().st_size, 'sha256': wan_assets.sha256_file(array)}
    meta = tmp_path / 'text_embeddings_meta.json'
    meta.write_text(json.dumps(header))
    before = (meta.read_bytes(), array.read_bytes())
    def unexpected_encoder(*args):
        raise AssertionError('existing cache must be reused')
    monkeypatch.setattr(wan_assets, '_load_text_encoder', unexpected_encoder)
    kwargs = dict(model_root=tmp_path, source_repo=tmp_path, device=torch.device('cpu'),
                  asset_receipt_sha256='a' * 64, expected_sequence_length=4)
    wan_assets.prepare_wan_text_embeddings(tmp_path, tasks, **kwargs)
    assert (meta.read_bytes(), array.read_bytes()) == before
    header['format'] = 'unrecognized-cache'
    meta.write_text(json.dumps(header))
    with pytest.raises(ValueError, match='contract differs'):
        wan_assets.prepare_wan_text_embeddings(tmp_path, tasks, **kwargs)


def test_new_text_cache_uses_current_name(monkeypatch, tmp_path):
    def encoder(*args):
        return lambda tasks, device: [torch.ones(2, 4096) for _ in tasks]
    monkeypatch.setattr(wan_assets, '_load_text_encoder', encoder)
    wan_assets.prepare_wan_text_embeddings(tmp_path, ['pick'], model_root=tmp_path,
        source_repo=tmp_path, device=torch.device('cpu'), asset_receipt_sha256='a' * 64)
    meta = json.loads((tmp_path / 'text_embeddings_meta.json').read_text())
    assert meta['format'] == 'zerith-umt5-text-embeddings-v1'
    np.testing.assert_array_equal(np.load(tmp_path / 'text_embeddings.npy'), np.ones((1, 2, 4096), dtype=np.float16))


@pytest.mark.parametrize('old,current', [
    ('zvla_semantic_camera3', 'zerith_semantic_camera3'),
    ('fam-wam-v5-cfs-asset-receipt-v1', 'zerith-asset-receipt-v1'),
    ('fam-wam-v5.2-node-local-assets-v1', 'zerith-node-local-assets-v1'),
])
def test_old_schema_identifiers_remain_readable(old, current):
    from zerith_sim.compat import same_identifier
    assert same_identifier(old, current)
    assert not same_identifier(old + '-unknown', current)
