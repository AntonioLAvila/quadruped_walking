from argparse import ArgumentParser
from pathlib import Path
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from util import get_run_id  

def _merge_datas(datas):
    master_data = {}
    for data in datas:
        for key, item in data.items():
            if key in master_data:
                print(f"[WARN] Duplicate run_id '{key}' while merging JSON DBs. Overwriting.")
            master_data[key] = item
    return master_data


def _fmt_alpha(alpha):
    if alpha is None:
        return "NA"
    return str(alpha).rstrip("0").rstrip(".") if isinstance(alpha, float) else str(alpha)

def _model_name_from_config(cfg):
    """
    Map config to a human-readable model_name:

    Baseline (no noise, no delay):
        noise_type=None, obs_delay=0 → baseline_h{h}

    Delay (no noise but with delay):
        noise_type=None, obs_delay>0 → delay{d}_h{h}

    Low-pass filter:
        noise_type=LPF → lpf_a{a}_h{h}

    High-pass filter:
        noise_type=HPF → hpf_a{a}_h{h}
    """
    noise_type = cfg.get("noise_type", "None")
    history_length = cfg.get("history_length", "NA")
    obs_delay = cfg.get("obs_delay", 0)
    alpha = cfg.get("alpha", None)

    a_str = _fmt_alpha(alpha)

    nt = str(noise_type).upper()

    # Delay and baseline
    if nt == "NONE":
        if obs_delay > 0:
            return f"delay{obs_delay}_h{history_length}"
        else:
            return f"baseline_h{history_length}"

    # LPF
    if nt == "LPF":
        return f"lpf_a{a_str}_h{history_length}"

    # HPF
    if nt == "HPF":
        return f"hpf_a{a_str}_h{history_length}"

    # Unknown
    return f"{noise_type}_h{history_length}_d{obs_delay}"


def _flatten_run(run_id, entry):
    """
    Flatten one run entry from the raw DB into a flat dict with keys matching the YAML.

    Raw structure:
    {
        "timestamp": "...",
        "config": {...},
        "metrics": {...}
    }
    """
    cfg = entry.get("config", {})
    met = entry.get("metrics", {})

    history_length = cfg.get("history_length")
    obs_delay = cfg.get("obs_delay", 0)
    noise_type = cfg.get("noise_type", "None")
    alpha = cfg.get("alpha", None)
    torque_scale = cfg.get("torque_scale", None)

    total_reward = met.get("total_reward", None)
    total_length = met.get("total_length", None)
    falls = met.get("falls", None)

    avg_velocity = met.get("avg_velocity", [0.0, 0.0, 0.0])
    avg_angular_velocity = met.get("avg_angular_velocity", [0.0, 0.0, 0.0])
    avg_gravity_projection = met.get("avg_gravity_projection", [0.0, 0.0, 0.0])

    avg_velocity_magnitude = float(np.linalg.norm(avg_velocity))
    avg_angular_velocity_magnitude = float(np.linalg.norm(avg_angular_velocity))
    avg_gravity_projection_z = float(avg_gravity_projection[2])
    avg_torque_norm = met.get("avg_torque_norm", None)
    avg_power = met.get("avg_power", None)

    model_name = _model_name_from_config(cfg)

    flat = {
        "run_id": run_id,
        "model_name": model_name,
        "history_length": history_length,
        "obs_delay": obs_delay,
        "noise_type": noise_type,
        "alpha": alpha,
        "torque_scale": torque_scale,
        # for YAML fields not in model:
        "total_reward": total_reward,
        "total_length": total_length,
        "falls": falls,
        "avg_velocity_magnitude": avg_velocity_magnitude,
        "avg_angular_velocity_magnitude": avg_angular_velocity_magnitude,
        "avg_gravity_projection_z": avg_gravity_projection_z,
        "avg_torque_norm": avg_torque_norm,
        "avg_power": avg_power,
    }
    return flat


def _flatten_master_data(master_raw):
    """
    Convert raw merged DB into {run_id: flat_record}.
    """
    flat = {}
    for run_id, entry in master_raw.items():
        flat[run_id] = _flatten_run(run_id, entry)
    return flat


def _get_numeric_keys(example_record, exclude=None):
    """
    Return all keys in example_record that are numeric (int/float),
    excluding any in `exclude`.
    """
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    numeric_keys = []
    for k, v in example_record.items():
        if k in exclude:
            continue
        if isinstance(v, (int, float)):
            numeric_keys.append(k)
    return numeric_keys


def swarm_graphing(job, flat_data):
    assert job.get("type", "") == "swarm_plot"
    figures = {}

    x_axes = job.get("x_axes", [])
    y_axes = job.get("y_axes", [])
    label_args = job.get("label_args", [])

    if not flat_data:
        print("[WARN] No data available for swarm_plot.")
        return figures

    runs = list(flat_data.values())
    example = runs[0]

    # use first label_arg as group key (e.g. history_length)
    if not label_args:
        print("[WARN] swarm_plot requires at least one label_arg to group by. Skipping.")
        return figures
    group_key = label_args[0]

    if group_key not in example:
        print(f"[WARN] swarm_plot group_key '{group_key}' not found in data. Skipping.")
        return figures

    # if y_axes == ["all"], interpret metrics as x_axes list from YAML.
    if len(y_axes) == 1 and y_axes[0] == "all":
        metrics = x_axes
    else:
        metrics = y_axes

    for metric in metrics:
        if metric not in example:
            print(f"[WARN] swarm_plot metric '{metric}' not found in data. Skipping.")
            continue

        grouped_values = {}
        for r in runs:
            if metric not in r or group_key not in r:
                continue
            gval = r[group_key]
            grouped_values.setdefault(gval, []).append(r[metric])

        if not grouped_values:
            print(f"[WARN] No data for swarm_plot metric '{metric}'. Skipping.")
            continue

        fig, ax = plt.subplots()
        xs, ys = [], []
        xticks = []
        xlabels = []

        for i, (gval, vals) in enumerate(sorted(grouped_values.items(), key=lambda kv: kv[0])):
            xticks.append(i)
            xlabels.append(str(gval))
            for v in vals:
                xs.append(i + (np.random.rand() - 0.5) * 0.3)
                ys.append(v)

        ax.scatter(xs, ys, alpha=0.7)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(group_key)
        ax.set_ylabel(metric)
        ax.set_title(f"Swarm plot of {metric} grouped by {group_key}")

        fig_name = f"swarm_{metric}_by_{group_key}.png"
        figures[fig_name] = fig

    return figures


def scatter_graphing(job, flat_data):
    assert job.get("type", "") == "scatter_plot"
    figures = {}

    x_axes = job.get("x_axes", [])
    y_axes = job.get("y_axes", [])
    label_args = job.get("label_args", [])

    if not flat_data:
        print("[WARN] No data available for scatter_plot.")
        return figures

    runs = list(flat_data.values())
    example = runs[0]

    # if y_axes == ["all"], interpret as all numeric metrics excluding label_args
    if len(y_axes) == 1 and y_axes[0] == "all":
        y_metrics = _get_numeric_keys(example, exclude=label_args)
    else:
        y_metrics = y_axes

    for x_name in x_axes:
        if x_name not in example:
            print(f"[WARN] scatter_plot x-axis '{x_name}' not found in data. Skipping.")
            continue

        for y_name in y_metrics:
            if y_name not in example:
                print(f"[WARN] scatter_plot y-axis '{y_name}' not found in data. Skipping.")
                continue
            if x_name == y_name:
                continue

            xs, ys, labels = [], [], []
            for r in runs:
                if x_name not in r or y_name not in r:
                    continue
                xs.append(r[x_name])
                ys.append(r[y_name])

                parts = []
                for la in label_args:
                    if la in r:
                        parts.append(f"{la}={r[la]}")
                labels.append(", ".join(parts))

            if not xs:
                print(f"[WARN] No valid points for scatter_plot {y_name} vs {x_name}. Skipping.")
                continue

            fig, ax = plt.subplots()
            ax.scatter(xs, ys)

            for x, y, lbl in zip(xs, ys, labels):
                if lbl:
                    ax.annotate(lbl, (x, y), fontsize=6, alpha=0.7)

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(f"{y_name} vs {x_name}")

            fig_name = f"scatter_{y_name}_vs_{x_name}.png"
            figures[fig_name] = fig

    return figures
def table_graph(job, flat_data):
    assert job.get("type", "") == "table_plot"
    figures = {}

    x_axes = job.get("x_axes", [])
    y_axes = job.get("y_axes", [])
    bold_rules = job.get("bold_rules", {})      # e.g. {"falls": "min", "avg_velocity_magnitude": "max"}
    color_rules = job.get("color_rules", {})    # e.g. {"falls": {"mode": "min", "scheme": "max_green_min_red"}, ...}

    if not flat_data:
        print("[WARN] No data available for table_plot.")
        return figures

    runs = list(flat_data.values())
    example = runs[0]

    if not y_axes:
        print("[WARN] table_plot requires y_axes (e.g. ['model_name']). Skipping.")
        return figures

    row_label_key = y_axes[0]
    if row_label_key not in example:
        print(f"[WARN] table_plot row label key '{row_label_key}' not in data. Skipping.")
        return figures

    # ---- Build raw numeric + formatted display data ----
    col_labels = [row_label_key] + x_axes

    numeric_data = []  # same shape as table data, but only numeric values (or None)
    display_data = []  # what we actually show in the table

    for r in runs:
        row_label = r.get(row_label_key, "")
        row_numeric = [None]          # first column is label, not numeric
        row_display = [row_label]

        for k in x_axes:
            v = r.get(k, None)
            if isinstance(v, (int, float)):
                row_numeric.append(v)
                row_display.append(f"{v:.3f}")   # 3 decimal places
            else:
                row_numeric.append(None)
                row_display.append("" if v is None else str(v))

        numeric_data.append(row_numeric)
        display_data.append(row_display)

    # ---- Compute per-column min/max on numeric_data ----
    col_stats = {}  # col_index -> {"min": ..., "max": ...}
    num_cols = len(x_axes) + 1  # including label col at index 0

    for j in range(1, num_cols):  # j=0 is model_name
        vals = [row[j] for row in numeric_data if isinstance(row[j], (int, float))]
        if not vals:
            continue
        col_stats[j] = {"min": min(vals), "max": max(vals)}

    # ---- Create figure + table ----
    fig_width = max(4, len(col_labels) * 1.2)
    fig_height = max(4, len(display_data) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(cellText=display_data, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    # Make columns wide enough for header text
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # ---- Apply bolding (per-column rules) ----
    for i, row in enumerate(numeric_data):      # i = row index in data
        for j, cell_val in enumerate(row):      # j = col index
            if j == 0:
                continue  # skip model_name column

            col_name = x_axes[j - 1]
            rule = bold_rules.get(col_name, None)   # "min", "max", "both", or None
            if rule not in ["min", "max", "both"]:
                continue

            if not isinstance(cell_val, (int, float)):
                continue

            stats = col_stats.get(j, None)
            if not stats:
                continue

            is_min = (cell_val == stats["min"])
            is_max = (cell_val == stats["max"])

            should_bold = (
                (rule == "min" and is_min) or
                (rule == "max" and is_max) or
                (rule == "both" and (is_min or is_max))
            )

            if should_bold:
                cell = table[i + 1, j]  # +1 because row 0 is header
                cell.get_text().set_fontweight("bold")

    # ---- Apply color highlighting (per-column rules + per-column scheme) ----
    for i, row in enumerate(numeric_data):
        for j, cell_val in enumerate(row):
            if j == 0:
                continue  # skip model_name

            col_name = x_axes[j - 1]
            rule_cfg = color_rules.get(col_name, None)
            if rule_cfg is None:
                continue

            # Accept both simple form ("min"/"max") and dict form.
            if isinstance(rule_cfg, str):
                mode = rule_cfg
                scheme = "max_green_min_red"
            else:
                mode = rule_cfg.get("mode", "max")       # "min", "max", "both"
                scheme = rule_cfg.get("scheme", "max_green_min_red")

            if mode not in ["min", "max", "both"]:
                continue

            if not isinstance(cell_val, (int, float)):
                continue

            stats = col_stats.get(j, None)
            if not stats:
                continue

            is_min = (cell_val == stats["min"])
            is_max = (cell_val == stats["max"])

            # Determine colors for this column
            if scheme == "max_green_min_red":
                max_color = "#b6f5b6"   # light green
                min_color = "#f7b5b5"   # light red
            elif scheme == "max_red_min_green":
                max_color = "#f7b5b5"
                min_color = "#b6f5b6"
            elif scheme == "min_only":
                max_color = None
                min_color = "#f7b5b5"
            elif scheme == "max_only":
                max_color = "#b6f5b6"
                min_color = None
            else:
                max_color = "#b6f5b6"
                min_color = "#f7b5b5"

            cell = table[i + 1, j]

            if is_min and mode in ["min", "both"] and min_color is not None:
                cell.set_facecolor(min_color)
            if is_max and mode in ["max", "both"] and max_color is not None:
                cell.set_facecolor(max_color)

    fig.tight_layout()
    fig_name = "table_plot.png"
    figures[fig_name] = fig
    return figures


def graph_queue(args):
    paths_json_db = [Path(f).expanduser() for f in args.paths_json_db]
    path_graphing_yaml = Path(args.path_graphing_yaml).expanduser()
    dir_out = Path(args.dir_out).expanduser() if args.dir_out else Path(".")
    dir_out.mkdir(parents=True, exist_ok=True)

    for path_json_db in paths_json_db:
        assert path_json_db.exists(), f"JSON DB path does not exist: {path_json_db}"
    assert path_graphing_yaml.exists(), f"Graphing YAML path does not exist: {path_graphing_yaml}"

    datas = []
    for path_json_db in paths_json_db:
        with open(path_json_db, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"JSON DB at {path_json_db} is not a dict."
            datas.append(data)

    assert len(datas) > 0, "No JSON DBs loaded."

    master_raw = _merge_datas(datas)

    if args.path_json_db_out:
        path_json_db_out = Path(args.path_json_db_out).expanduser()
        path_json_db_out.parent.mkdir(parents=True, exist_ok=True)
        with open(path_json_db_out, "w") as f:
            json.dump(master_raw, f, indent=4)
        print(f"Combined raw JSON DB written to {path_json_db_out}")

    flat_data = _flatten_master_data(master_raw)

    # load graph yaml config
    with open(path_graphing_yaml, "r") as f:
        config = yaml.safe_load(f)

    jobs = config.get("graph_jobs", [])
    assert len(jobs) > 0, "No graph_jobs found in YAML."

    # create graphs
    figures = {}  # fig_name -> fig
    for job in jobs:
        graph_type = job.get("type", None)
        assert graph_type is not None, "Every job must have a 'type' field."

        if graph_type == "scatter_plot":
            figs = scatter_graphing(job, flat_data)
        elif graph_type == "swarm_plot":
            figs = swarm_graphing(job, flat_data)
        elif graph_type == "table_plot":
            figs = table_graph(job, flat_data)
        else:
            raise TypeError(f"Graph type '{graph_type}' is not recognized!")

        for name, fig in figs.items():
            if name in figures:
                print(f"[WARN] Figure name collision for '{name}'. Overwriting previous figure.")
            figures[name] = fig

    for plt_name, fig in figures.items():
        path_plt = dir_out / plt_name
        fig.savefig(path_plt, bbox_inches="tight")
        print(f"Saved figure: {path_plt}")

    print("\nAll graphing jobs completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--paths_json_db",
        type=str,
        nargs="*",
        required=True,
        help="Paths to json db",
    )
    parser.add_argument(
        "--path_graphing_yaml",
        type=str,
        required=True,
        help="Path to yaml for graphing jobs",
    )

    parser.add_argument(
        "--path_json_db_out",
        type=str,
        required=False,
        default="",
        help="Path to save a json that's a combination of all input jsons (raw format).",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=False,
        default="",
        help="Directory to output graphics (defaults to current directory)",
    )
    args = parser.parse_args()

    graph_queue(args)
