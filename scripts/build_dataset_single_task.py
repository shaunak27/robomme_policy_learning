"""Build preprocessed dataset for a single h5 file.

Writes to a task-specific output directory to avoid collisions when
running multiple tasks in parallel. Use scripts/merge_task_datasets.py
to combine them afterwards.

Usage:
    python scripts/build_dataset_single_task.py \
        --h5_file data/robomme_data_h5/record_dataset_BinFill.h5 \
        --output_dir data/robomme_preprocessed_data/per_task/BinFill
"""

import argparse
import time

from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to a single .h5 file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Task-specific output directory")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    import os
    import shutil
    import tempfile

    h5_file = os.path.abspath(args.h5_file)

    # Use a temp dir outside the output path for the symlink, because
    # DatasetProcessor.__init__ does rmtree on its output dir.
    tmp_raw_dir = tempfile.mkdtemp(prefix="build_dataset_")
    link_path = os.path.join(tmp_raw_dir, os.path.basename(h5_file))
    os.symlink(h5_file, link_path)

    t0 = time.perf_counter()
    processor = DatasetProcessor(
        raw_data_path=tmp_raw_dir,
        preprocessed_data_path=args.output_dir,
        visualize=args.visualize,
        max_episodes=args.max_episodes,
    )
    processor.run()

    # Clean up temp symlink dir
    shutil.rmtree(tmp_raw_dir, ignore_errors=True)

    print(f"Done in {(time.perf_counter() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
