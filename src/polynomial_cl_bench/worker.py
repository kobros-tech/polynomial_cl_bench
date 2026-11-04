# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""One benchmark run in a fresh process: ``python -m polynomial_cl_bench.worker task.json out.json``.

The runner starts this for every algorithm x seed under ``isolation: process``. It loads and
verifies the data, builds the shared plan, executes the run with the same ``execute_run`` used
in-process, and writes the validated record. Exit codes: 0 record written (a failed *run* is
still a written record); 3 official-mode provenance refusal; anything else: setup crash (the
runner records it as ``crashed`` with this process's stderr).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import results
from .protocol import Protocol
from .provenance import ProvenanceError


def main(argv=None) -> int:
    task_path, output_path = (argv if argv is not None else sys.argv[1:])[:2]
    try:
        task = json.loads(Path(task_path).read_text())
        from .datasets import load_split_mnist, validation_view
        from .plan import make_plan
        from .runner import build_adapter, execute_run

        protocol = Protocol.from_dict(task["protocol"])
        dataset = load_split_mnist(protocol, task["data_dir"])
        if task["evaluation_split"] == "validation":
            dataset = validation_view(dataset, protocol)
        plan = make_plan(dataset.train_y, task["seed"], protocol.memory_per_class)
        record = execute_run(
            label=task["label"],
            config=task["config"],
            protocol=protocol,
            dataset=dataset,
            seed=task["seed"],
            plan=plan,
            mode=task["mode"],
            attempt=task["attempt"],
            isolation="process",
            experiment=task.get("experiment"),
            adapter_factory=build_adapter,
        )
        results.write_json_atomic(output_path, record)
        return 0
    except ProvenanceError as error:
        print(str(error), file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - reported to the parent through stderr and exit code
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
