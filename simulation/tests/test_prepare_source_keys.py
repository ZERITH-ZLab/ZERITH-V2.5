import io
import struct

import numpy as np
from PIL import Image
from zerith_sim.posttrain.sources import libero_episodes
from zerith_sim.posttrain.tfrecord import TF_EXAMPLE


def jpeg(color):
    encoded = io.BytesIO()
    Image.new('RGB', (7, 5), color).save(encoded, format='JPEG', quality=100, subsampling=0)
    return encoded.getvalue()


def masked_crc32c(value):
    crc = 0xffffffff
    for byte in value:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    crc ^= 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def tfrecord(example):
    payload = example.SerializeToString()
    size = struct.pack('<Q', len(payload))
    return size + struct.pack('<I', masked_crc32c(size)) + payload + struct.pack('<I', masked_crc32c(payload))


def test_libero_episode_keys_distinguish_same_filename_in_different_directories(tmp_path):
    example = TF_EXAMPLE()
    features = example.features.feature
    features['steps/observation/state'].float_list.value.extend([.1, .2, .3, 0, 0, 0, .04, 0])
    features['steps/action'].float_list.value.extend([0] * 7)
    features['steps/language_instruction'].bytes_list.value.append(b'move the block')
    for camera in ('image', 'wrist_image'):
        features['steps/observation/' + camera].bytes_list.value.append(jpeg((180, 50, 20)))
    filename = 'train.tfrecord-00000-of-00001'
    for task in ('task_a', 'task_b'):
        directory = tmp_path / task
        directory.mkdir()
        (directory / filename).write_bytes(tfrecord(example))

    records = list(libero_episodes(tmp_path))
    assert [row[0] for row in records] == [f'{task}/{filename}-0' for task in ('task_a', 'task_b')]
    assert len({row[0] for row in records}) == 2
    for _, prompts, state, target, sv, av, decode in records:
        assert prompts == ['move the block']
        assert state.shape == target.shape == sv.shape == av.shape == (1, 20)
        streams = decode()
        assert len(streams) == 2
        for frames in streams:
            assert frames.shape == (1, 5, 7, 3)
            assert frames.dtype == np.uint8
            np.testing.assert_allclose(frames[0, 0, 0], [180, 50, 20], atol=2)
