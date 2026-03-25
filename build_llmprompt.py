# build_llm_prompt.py (fixed paths version, 80 episodes for labeled & query)
import os, json
from collections import defaultdict
from typing import Dict, Any, List, Optional

# ==== Fixed paths ====
TASK1_FILE = "collectdata/train_traj_task1.jsonl"
TASK2_FILE = "collectdata/train_traj_task2.jsonl"
TASK3_FILE = "collectdata/train_traj_task3.jsonl"
QUERY_FILE = "collectdata/query_traj_1.jsonl"   # choose which file to classify
OUT_FILE   = "collectdata/prompt_en.txt"

# ==== Sampling & limits ====
SAMPLE_INTERVAL = 8              # sample every 8 steps
EPISODES_PER_TASK = 80           # keep 80 episodes per labeled task
QUERY_EPISODES = 80              # keep 80 episodes for the unlabeled query
MAX_POINTS_PER_EPISODE = 6       # cap sampled points per episode

TASK_NAME = {
    1: "Task 1 (Minimize voltage fluctuation)",
    2: "Task 2 (Minimize total operating cost)",
    3: "Task 3 (Maximize PV utilization)",
}

# keys extracted from each JSONL record (features.* paths map to compact names)
KEYS = [
    ("t", "t"),
    ("reward", "reward"),
    ("voltage.mean_abs_dev", "v_mean_abs_dev"),
    ("voltage.max_abs_dev",  "v_max_abs_dev"),
    ("cost.P_grid_proxy",    "P_grid"),
    ("cost.cum_cost_proxy",  "cum_cost"),
    ("pv.util",              "pv_util"),
]

def get_nested(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def load_and_sample(jsonl_path: str,
                    sample_interval: int,
                    episodes_limit: int,
                    max_points_per_ep: int) -> Dict[int, List[Dict[str, Any]]]:
    """Read JSONL, group by episode, downsample by interval, keep key fields."""
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    ep_data: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec.get("episode", 0))
            t  = int(rec.get("t", 0))

            # sample every N steps
            if (t % max(1, sample_interval)) != 0:
                continue

            out_item = {"t": t, "reward": float(rec.get("reward", 0.0))}
            feat = rec.get("features", {})
            for src, dst in KEYS:
                if src in ("t", "reward"):
                    continue
                out_item[dst] = get_nested(feat, src, None)
            ep_data[ep].append(out_item)

    # per-episode cap & sort by time
    for ep in ep_data:
        ep_data[ep].sort(key=lambda x: x.get("t", 0))
        if max_points_per_ep > 0:
            ep_data[ep] = ep_data[ep][:max_points_per_ep]

    # keep only first `episodes_limit` episodes (by episode id)
    if episodes_limit > 0:
        selected: Dict[int, List[Dict[str, Any]]] = {}
        for ep in sorted(ep_data.keys()):
            selected[ep] = ep_data[ep]
            if len(selected) >= episodes_limit:
                break
        ep_data = selected

    return ep_data

def _fnum(x, nd=4, fallback="NA"):
    """safe number formatting for possibly None values"""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return fallback

def snippets_to_text(snips: Dict[int, List[Dict[str, Any]]], label: Optional[int]) -> str:
    """Render sampled snippets into compact, human-readable text."""
    lines: List[str] = []
    if label is not None:
        lines.append(f"### Labeled samples for {TASK_NAME[label]}")
    else:
        lines.append("### Unlabeled trajectory (classify this one)")

    for ep in sorted(snips):
        lines.append(f"- episode = {ep}")
        for pt in snips[ep]:
            lines.append(
                "  t={t}, R={R}, v_mean_abs_dev={vmad}, v_max_abs_dev={vmax}, "
                "P_grid={pgrid}, cum_cost={ccost}, pv_util={pv}".format(
                    t=pt.get("t"),
                    R=_fnum(pt.get("reward"), nd=3),
                    vmad=_fnum(pt.get("v_mean_abs_dev")),
                    vmax=_fnum(pt.get("v_max_abs_dev")),
                    pgrid=_fnum(pt.get("P_grid"), nd=3),
                    ccost=_fnum(pt.get("cum_cost"), nd=3),
                    pv=_fnum(pt.get("pv_util"), nd=3),
                )
            )
    lines.append("")  # blank line
    return "\n".join(lines)

def build_prompt() -> str:
    # load & downsample
    t1 = load_and_sample(TASK1_FILE, SAMPLE_INTERVAL, EPISODES_PER_TASK, MAX_POINTS_PER_EPISODE)
    t2 = load_and_sample(TASK2_FILE, SAMPLE_INTERVAL, EPISODES_PER_TASK, MAX_POINTS_PER_EPISODE)
    t3 = load_and_sample(TASK3_FILE, SAMPLE_INTERVAL, EPISODES_PER_TASK, MAX_POINTS_PER_EPISODE)
    q  = load_and_sample(QUERY_FILE,  SAMPLE_INTERVAL, QUERY_EPISODES,     MAX_POINTS_PER_EPISODE)

    header = f"""# Multi-Task Routing Prompt (English)

You are given three optimization objectives for distribution system control:

- {TASK_NAME[1]}
- {TASK_NAME[2]}
- {TASK_NAME[3]}

We provide **condensed training trajectory snippets** (NOT full logs) per task.
We **sample every {SAMPLE_INTERVAL} steps** within each episode and only keep key fields:

Field definitions per sampled step:
- `t`: time step index within the episode.
- `R` (reward): the scalar reward at step `t`.
- `v_mean_abs_dev`: average absolute voltage deviation |V-1.0| across all buses; lower is better for Task 1.
- `v_max_abs_dev`: maximum absolute voltage deviation across all buses; lower reduces worst-case violations.
- `P_grid`: proxy of purchased active power from the grid (positive ≈ importing).
- `cum_cost`: cumulative cost proxy up to this step within the current episode; lower is better for Task 2.
- `pv_util`: PV utilization ratio, i.e., (available PV - curtailed PV) / available PV; higher is better for Task 3.

**Heuristic cues to distinguish tasks:**
- **Task 1 (voltage)**: monotonic or consistent reduction in `v_mean_abs_dev` / `v_max_abs_dev`.
- **Task 2 (cost)**: decreasing `cum_cost` growth rate and/or reduced `P_grid` especially during expensive periods.
- **Task 3 (PV)**: higher `pv_util` with lower curtailment whenever PV is available.

Below are labeled snippets per task (up to {EPISODES_PER_TASK} episodes each), followed by one **unlabeled** snippet to classify (up to {QUERY_EPISODES} episodes).
Each line shows: `t, R, v_mean_abs_dev, v_max_abs_dev, P_grid, cum_cost, pv_util`.
"""

    body: List[str] = []
    body.append(snippets_to_text(t1, 1))
    body.append(snippets_to_text(t2, 2))
    body.append(snippets_to_text(t3, 3))
    body.append(snippets_to_text(q, None))

    tail = """## Output format (STRICT)
Return ONLY one of the following single digits:
- `1`  (Task 1: Minimize voltage fluctuation)
- `2`  (Task 2: Minimize total operating cost)
- `3`  (Task 3: Maximize PV utilization)

Do not include any explanations or extra symbols. Output a single digit only.
"""
    return header + "\n".join(body) + tail

if __name__ == "__main__":
    prompt = build_prompt()
    os.makedirs(os.path.dirname(os.path.abspath(OUT_FILE)), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"✅ Prompt saved to {OUT_FILE}")
