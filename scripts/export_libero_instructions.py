"""Export the 40 base task instructions from a standard LIBERO source checkout.

Only the literal task map is read; LIBERO, PyTorch and simulator packages are
neither imported nor required. LIBERO-Plus task maps are rejected.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def export_instructions(libero_root: Path, output: Path) -> list[str]:
    source = libero_root / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(source.read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("Expected the standard LIBERO literal task map")
    task_map = ast.literal_eval(assignments[0].value)
    instructions = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        tasks = task_map[suite]
        if len(tasks) != 10 or any(not isinstance(task, str) for task in tasks):
            raise ValueError("Use a standard LIBERO checkout, not LIBERO-Plus")
        instructions.extend(re.sub(r"^.*?SCENE\d+_", "", task).replace("_", " ") for task in tasks)
    if len(set(instructions)) != 40:
        raise ValueError("Expected 40 unique standard LIBERO instructions")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(instructions, indent=2) + "\n")
    return instructions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = export_instructions(args.libero_root, args.output)
    print(f"Exported {len(values)} base instructions to {args.output}")


if __name__ == "__main__":
    main()
