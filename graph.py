from argparse import ArgumentParser
from tqdm import tqdm
import time
import numpy as np
import json
from pathlib import Path
from util import get_run_id
import matplotlib.pyplot as plt
import yaml


def swarm_graphing(job, data):
    assert job.get("type", "") == "swarm_plot"
    names_to_figures = dict()

    x_axes = job["x_axes"]
    print("SWARM")
    print(x_axes)
    # for 

def scatter_graphing(job, data):
    print("SCATTER")
    pass
    assert job.get("type", "") == "scatter_plot"


def graph_queue(args):
    # checks
    path_json_db = Path(args.path_json_db)
    path_graphing_yaml = Path(args.path_graphing_yaml)
    dir_out = Path(args.dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    assert path_json_db.exists()
    assert path_graphing_yaml.exists()
    
    # get db + config
    with open(path_json_db, 'r') as f:
        data = json.load(f)
    with open(path_json_db, 'r') as f:
        config = yaml.safe_load(f)
    jobs = config.get("graph_jobs", [])
    assert isinstance(data, dict)
    assert len(jobs) > 0

    # create all graphs
    jobs_to_fig = dict() # index of job to (fig_name, fig)
    for i, job in enumerate(jobs):
        graph_type = jobs.get("type", None)
        assert graph_type is not None
        match graph_type:
            case "scatter_plot":
                jobs_to_fig[i] = scatter_graphing(job)
            case "swarm_plot":
                jobs_to_fig[i] = swarm_graphing(job)
            case default:
                assert TypeError(f"graph type {default} is not recognised!")

    # save all graphs
    for i_job, (plt_name, fig) in jobs_to_fig.items():
        path_plt = dir_out / plt_name
        fig.savefig(path_plt)

    print("\nAll graphing jobs completed successfully.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_json_db", type=str, required=True, help="Path to json db")
    parser.add_argument("--path_graphing_yaml", type=str, required=True, help="Path to yaml for graphing jobs")

    parser.add_argument("--dir_out", type=str, required=False, default="", help="Enter directory to output graphics")    
    args = parser.parse_args()

    graph_queue(args)
