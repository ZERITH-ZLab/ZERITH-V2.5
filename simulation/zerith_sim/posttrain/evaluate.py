"""Benchmark rollouts using the exported HTTP policy."""

import json
import sys
from pathlib import Path

import numpy as np

from .client import PolicyClient
from .recipes import BENCHMARKS


def _libero_dependencies(bench):
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import quat2axisangle
    except ImportError as error:
        raise ImportError(
            f"{bench} evaluation requires its official LIBERO/robosuite environment. "
            "Install the selected benchmark's dependencies using its official setup "
            f"instructions in this environment; missing or incompatible import: {error}"
        ) from error
    return benchmark, get_libero_path, OffScreenRenderEnv, quat2axisangle


def _metaworld_dependency():
    try:
        import metaworld
    except ImportError as error:
        raise ImportError(
            "metaworld evaluation requires its official MetaWorld/MuJoCo environment. "
            "Install the selected benchmark's dependencies using its official setup "
            f"instructions in this environment; missing or incompatible import: {error}"
        ) from error
    return metaworld


def _benchmark_root(value, *, required=False):
    if value is None or value == "":
        if required:
            raise ValueError("Set --benchmark-root to the selected benchmark's upstream checkout")
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"--benchmark-root must be an existing upstream directory: {root}")
    return root


def libero(args):
    benchmark, get_libero_path, OffScreenRenderEnv, quat2axisangle = _libero_dependencies(
        args.bench
    )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = getattr(args, "task_ids", None)
    if task_ids is None:
        task_ids = range(suite.n_tasks)
    elif any(task_id >= suite.n_tasks for task_id in task_ids):
        raise ValueError(f"--task-ids must be smaller than {suite.n_tasks} for {args.suite}")
    horizon = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[
        args.suite
    ]
    client = PolicyClient(args.server)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            starts = suite.get_task_init_states(task_id)
            for episode in range(args.episodes):
                env.seed(args.seed + episode)
                env.reset()
                obs = env.set_init_state(starts[episode])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                client.reset(task.language, args.seed + episode)
                frames, actions, success = [], [], False
                for step in range(horizon):  # noqa: B007 - report the final step after rollout
                    frames.append(
                        np.stack(
                            [
                                obs["agentview_image"][::-1, ::-1],
                                obs["robot0_eye_in_hand_image"][::-1, ::-1],
                            ]
                        )
                    )
                    if not len(actions):
                        state = np.r_[
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]
                        actions = list(client.infer(np.stack(frames), state))
                        frames = []
                    action = actions.pop(0)
                    # Model/native training convention: +1 closes, -1 opens.
                    action[6] = 1 if action[6] >= 0 else -1
                    obs, _, done, _ = env.step(action)
                    if done:
                        success = True
                        break
                yield dict(task=task_id, episode=episode, success=success, steps=step + 1)
        finally:
            env.close()


def metaworld(args):
    metaworld = _metaworld_dependency()
    client = PolicyClient(args.server)
    mt = metaworld.MT50(seed=args.seed)
    for name, cls in mt.train_classes.items():
        env = cls(render_mode="rgb_array", width=480, height=480, camera_name="corner2")
        goals = [task for task in mt.train_tasks if task.env_name == name]
        try:
            for episode in range(args.episodes):
                env.set_task(goals[episode])
                obs, _ = env.reset(seed=args.seed + episode)
                from metaworld_benchmark.prompts import PROMPTS

                prompt = PROMPTS[name]
                client.reset(prompt, args.seed + episode)
                frames, actions, success = [], [], False
                for step in range(500):  # noqa: B007 - report the final step after rollout
                    frames.append(env.render()[None])
                    if not len(actions):
                        actions = list(client.infer(np.stack(frames), obs[:4]))
                        frames = []
                    obs, _, terminated, truncated, info = env.step(actions.pop(0))
                    success |= bool(info["success"])
                    if terminated or truncated:
                        break
                yield dict(task=name, episode=episode, success=success, steps=step + 1)
        finally:
            env.close()


def evaluate(args):
    if args.bench not in BENCHMARKS:
        raise ValueError(f"Unsupported evaluation benchmark: {args.bench}")
    task_ids = getattr(args, "task_ids", None)
    if task_ids is not None:
        if args.bench == "metaworld":
            raise ValueError("--task-ids is supported only for LIBERO and LIBERO-Plus")
        if not task_ids or any(task_id < 0 for task_id in task_ids):
            raise ValueError("--task-ids must contain nonnegative task IDs")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("--task-ids must not contain duplicates")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    root = _benchmark_root(args.benchmark_root)
    if root is not None:
        sys.path.insert(0, str(root))
    if args.bench == "metaworld":
        _metaworld_dependency()
    else:
        _libero_dependencies(args.bench)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rollout = metaworld if args.bench == "metaworld" else libero
    # A failed/interrupted run must not leave an earlier completed summary behind.
    summary_path = output / "summary.json"
    summary_path.unlink(missing_ok=True)
    completed, successes = 0, 0
    with (output / "episodes.jsonl").open("w") as handle:
        for row in rollout(args):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(row, flush=True)
            completed += 1
            successes += int(row["success"])
    summary = dict(
        benchmark=args.bench,
        suite=args.suite if args.bench != "metaworld" else "MT50",
        task_ids=task_ids,
        seed=args.seed,
        episodes_per_task=args.episodes,
        completed_episodes=completed,
        successes=successes,
        success_rate=successes / completed if completed else None,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("EVAL_SUMMARY", json.dumps(summary), flush=True)
