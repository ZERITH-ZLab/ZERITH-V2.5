"""Simulator clients import only their selected benchmark dependencies."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zerith_sim.posttrain import evaluate as entrypoint
from zerith_sim.posttrain.recipes import BENCHMARKS
from zerith_sim.posttrain.sources import convert_rows, episodes


def test_libero_task_selection_drives_only_requested_rollouts(monkeypatch):
    opened, closed = [], []
    suite = SimpleNamespace(
        n_tasks=100,
        get_task=lambda i: SimpleNamespace(language=f"task {i}", problem_folder="suite", bddl_file=f"{i}.bddl"),
        get_task_init_states=lambda i: [i],
    )
    observation = dict(
        agentview_image=np.zeros((4, 4, 3), dtype=np.uint8),
        robot0_eye_in_hand_image=np.zeros((4, 4, 3), dtype=np.uint8),
        robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.zeros(4), robot0_gripper_qpos=np.zeros(2),
    )

    class Env:
        def __init__(self, bddl_file_name, **kwargs):
            self.task = int(Path(bddl_file_name).stem)
            opened.append(self.task)

        def seed(self, seed): pass
        def reset(self): pass
        def set_init_state(self, state): return observation
        def step(self, action): return observation, 1, True, {}
        def close(self): closed.append(self.task)

    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
    monkeypatch.setattr(entrypoint, "_libero_dependencies", lambda _: (benchmark, lambda _: "/bddl", Env, lambda _: np.zeros(3)))
    monkeypatch.setattr(entrypoint, "PolicyClient", lambda _: SimpleNamespace(
        reset=lambda *a: None, infer=lambda *a: np.zeros((4, 7)),
    ))
    args = SimpleNamespace(bench="libero-plus", suite="libero_spatial", task_ids=[30, 0, 60],
                           server="http://localhost", episodes=1, seed=0)
    rows = list(entrypoint.libero(args))
    assert [row["task"] for row in rows] == [30, 0, 60]
    assert all(row["success"] for row in rows)
    assert opened == closed == [30, 0, 60]
    args.task_ids = [100]
    with pytest.raises(ValueError, match="smaller than 100"):
        list(entrypoint.libero(args))
    assert opened == [30, 0, 60]
    opened.clear()
    closed.clear()
    suite.n_tasks = 3
    args.task_ids = None
    assert [row["task"] for row in entrypoint.libero(args)] == [0, 1, 2]
    assert opened == closed == [0, 1, 2]


@pytest.mark.parametrize("bench,task_ids,episodes", [
    ("libero-plus", [-1], 1), ("libero-plus", [0, 0], 1),
    ("metaworld", [0], 1), ("libero-plus", [0], 0),
])
def test_invalid_quick_evaluation_fails_before_simulator_imports(tmp_path, bench, task_ids, episodes, monkeypatch):
    monkeypatch.setattr(entrypoint, "_libero_dependencies", lambda _: pytest.fail("simulator imported"))
    args = SimpleNamespace(bench=bench, task_ids=task_ids, episodes=episodes, output=str(tmp_path / "out"))
    with pytest.raises(ValueError):
        entrypoint.evaluate(args)
    assert not (tmp_path / "out").exists()


def test_completed_and_interrupted_evaluation_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint, "_libero_dependencies", lambda _: None)
    rows = [dict(task=i, episode=0, success=success, steps=1)
            for i, success in zip([0, 30, 60], [True, False, True], strict=True)]
    monkeypatch.setattr(entrypoint, "libero", lambda _: iter(rows))
    args = SimpleNamespace(bench="libero-plus", task_ids=[0, 30, 60], episodes=1,
                           suite="libero_spatial", benchmark_root=None, seed=0, output=str(tmp_path))
    entrypoint.evaluate(args)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_episodes"] == 3 and summary["successes"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["task_ids"] == [0, 30, 60]
    assert [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()] == rows

    def interrupted(_):
        yield rows[0]
        raise RuntimeError("interrupted rollout")

    monkeypatch.setattr(entrypoint, "libero", interrupted)
    with pytest.raises(RuntimeError, match="interrupted rollout"):
        entrypoint.evaluate(args)
    assert not (tmp_path / "summary.json").exists()
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("bench", ["robotwin", "robodojo", "h1"])
def test_unsupported_simulation_targets_fail_before_side_effects(tmp_path, bench):
    with pytest.raises(ValueError, match="Unsupported evaluation benchmark"):
        entrypoint.evaluate(SimpleNamespace(bench=bench, output=str(tmp_path / "results")))
    with pytest.raises(ValueError, match="Unsupported preparation benchmark"):
        episodes(tmp_path, bench)
    with pytest.raises(ValueError, match="Unsupported preparation benchmark"):
        convert_rows(bench, [[0] * 20], [[0] * 20])
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("bench", BENCHMARKS)
def test_isolated_client_imports_only_the_selected_benchmark(tmp_path, bench):
    package = Path(entrypoint.__file__).resolve().parents[2]
    script = """
import builtins
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
blocked = {'libero', 'metaworld', 'robosuite', 'mujoco', 'torch'}
attempts = []
original_import = builtins.__import__
def client_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        attempts.append(name.split('.')[0])
        raise ModuleNotFoundError('Selected benchmark library is not installed', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = client_import
from zerith_sim.posttrain import evaluate as module
assert attempts == [], attempts
bench = sys.argv[2]
args = SimpleNamespace(bench=bench, benchmark_root=None,
    server='http://127.0.0.1:8000', output=sys.argv[3], seed=0,
    suite='libero_spatial', episodes=1)
try:
    module.evaluate(args)
except ImportError as error:
    assert bench in str(error) and 'official' in str(error), str(error)
else:
    raise AssertionError('Missing selected simulator dependencies were accepted')
assert attempts == ['metaworld' if bench == 'metaworld' else 'libero'], attempts
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package), bench, str(tmp_path / "results")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "results").exists()


def test_package_contains_supported_clients_and_extension_resources(monkeypatch):
    import setuptools

    package = Path(__file__).resolve().parents[1]
    captured = {}
    monkeypatch.chdir(package)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(package / "setup.py"))
    assert "metaworld_benchmark" in captured["packages"]
    assert "zerith_sim_runtime" in captured["packages"]
    assert "fa2_custom_mask" in captured["package_data"]
    assert not any(name.startswith(("robotwin", "robodojo")) for name in captured["packages"])
    assert captured["install_requires"] == []
    assert captured["extras_require"] == {"client": ["numpy>=1.26", "requests>=2.32"]}
