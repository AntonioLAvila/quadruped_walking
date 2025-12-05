import subprocess
import yaml
import sys
import os

import argparse

SCRIPT_MAP = {
    "train": "train.py",
    "test": "test.py",
}

def flag_from_bool(key, value):
    """
    Convert boolean args to CLI flags following your conventions.
    True  → "--key"
    False → "--no_key"
    """
    if value is True:
        return [f"--{key}"]
    return []


def build_cmd(job):
    job_type = job["type"]
    script = SCRIPT_MAP[job_type]
    args = job.get("args", {})

    cmd = ["python", script]

    for key, value in args.items():
        if isinstance(value, bool):
            cmd.extend(flag_from_bool(key, value))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    return cmd


def main(config_path):
    if not os.path.exists(config_path):
        print(f"ERROR: Could not find {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    jobs = config.get("jobs", [])
    print(jobs)
    if not jobs:
        print("No jobs found in YAML file.")
        return

    for i, job in enumerate(jobs, 1):
        cmd = build_cmd(job)

        print("\n" + "="*80)
        print(f"Starting job {i}/{len(jobs)}")
        print("Command:", " ".join(cmd))
        print("="*80)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Job {i} FAILED with return code {result.returncode}. Stopping.")
            return

    print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_yaml", type=str, required=True, help="path to yaml config job file")
    args = parser.parse_args()
    main(args.path_yaml)
