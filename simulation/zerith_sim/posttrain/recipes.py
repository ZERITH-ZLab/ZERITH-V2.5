"""Benchmark interfaces and suggested optimizer settings."""
from pathlib import Path

BENCHMARKS = {
    'libero': dict(robot='libero_franka', steps=30000, dt=.05, cameras=('front', 'wrist')),
    'libero-plus': dict(robot='libero_franka', steps=30000, dt=.05, cameras=('front', 'wrist')),
    'metaworld': dict(robot='metaworld_sawyer_applied4', steps=50000, dt=.0125, cameras=('corner2',)),
}


def recipe(bench, data_root='data/prepared', checkpoint=None, batch_size=10):
    spec = BENCHMARKS[bench]
    checkpoint = checkpoint or f"assets/policies/{bench}/model.safetensors"
    cfg = dict(
        model=dict(type='ZerithPostTrainModel', pretrained_checkpoint=str(checkpoint),
                   state_codec_stats=str(Path(data_root) / 'codec.json'),
                   config=dict(state_codec_embodiments=(spec['robot'],), training_source_names=(bench,),
                               metaworld_native4=bench == 'metaworld')),
        data=dict(train=dict(dataset=dict(type='BenchmarkDataset', data_root_path=str(data_root),
                    drop_last=False, sampler=dict(type='ZerithShardAwareBatchSampler', exact_once=True,
                                                 global_batch_mixing=True)),
                    collator=dict(type='ZerithStateTrajectoryGeometryRobotCollator'),
                    per_device_batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True)),
        distributed=dict(strategy='megatron_fsdp', mixed_precision='bf16'),
        runtime={},
        training=dict(run_name_prefix='zerith-'+bench, run_seed=0, learning_rate=1e-5,
            optimizer=dict(type='adamw', betas=(.9, .95)), weight_decay=.01, grad_clip=1.,
            max_steps=spec['steps'], grad_accumulation_steps=1,
            scheduler=dict(type='cosine', warmup_ratio=.05, min_lr_ratio=.1),
            checkpoint=dict(save_step_interval=1000, max_keep=2, save_full_model=True, format='sharded',
                            export_model_steps=(spec['steps'],), model_artifact_max_keep=2),
            metric=dict(type='VLAMetric', active_trackers=('jsonl',)),
            log_every_n_steps=50, source_metrics_every_n_steps=500, diagnostics_every_n_steps=1000,
            torch_compile=False))

    if bench == 'metaworld':
        cfg['data']['train']['dataset']['sampler'] = dict(type='EpisodeBalancedBatchSampler', steps=spec['steps'])
    return cfg
