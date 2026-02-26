"""
轨迹优化指标工具：输入优化前后轨迹（JSON/CSV，单条/多条），输出平滑度、位置偏差、移除帧比例。
各 metric 独立成函数，便于复用与测试。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# 末端偏差需 roboticstoolbox
try:
    import roboticstoolbox as rtb
    _RTB_AVAILABLE = True
except Exception:
    _RTB_AVAILABLE = False


# --------------- 数据加载 ---------------

def load_trajectories(path: str) -> list[dict[str, Any]]:
    """
    从 JSON 或 CSV 加载一条或多条轨迹。
    返回: list of {"meta": dict, "steps": list[{"simulation_time", "state": {"robot_joints": {...}}}}.
    """
    p = path.lower()
    if p.endswith(".json"):
        return _load_trajectories_json(path)
    if p.endswith(".csv"):
        return _load_trajectories_csv(path)
    raise ValueError(f"不支持的文件格式: {path}")


def _load_trajectories_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def extract_steps(obj):
        if not isinstance(obj, dict):
            return None
        td = obj.get("trajectory_data")
        if isinstance(td, dict) and "steps" in td:
            return td["steps"]
        return None

    out = []
    if isinstance(data, list):
        for item in data:
            steps = extract_steps(item)
            if steps is not None:
                out.append({"meta": item if isinstance(item, dict) else {}, "steps": steps})
    elif isinstance(data, dict):
        if "trajectories" in data:
            for item in data["trajectories"]:
                steps = extract_steps(item)
                if steps is not None:
                    out.append({"meta": item if isinstance(item, dict) else {}, "steps": steps})
        else:
            steps = extract_steps(data)
            if steps is not None:
                out.append({"meta": data, "steps": steps})
    if not out:
        raise ValueError("未找到有效的 trajectory_data.steps")
    return out


def _load_trajectories_csv(path: str) -> list[dict]:
    try:
        max_size = sys.maxsize
    except AttributeError:
        max_size = 2**31 - 1
    prev_limit = csv.field_size_limit()
    csv.field_size_limit(max_size)
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    finally:
        csv.field_size_limit(prev_limit)
    if not rows:
        raise ValueError("CSV 为空")
    first = rows[0]
    # 每行一条轨迹、trajectory_data 为 JSON 的格式
    if "trajectory_data" in first and isinstance(first.get("trajectory_data"), str) and first["trajectory_data"].strip().startswith("{"):
        out = []
        for row in rows:
            td = row.get("trajectory_data", "").strip()
            if not td.startswith("{"):
                continue
            try:
                data = json.loads(td)
                steps = data.get("steps") if isinstance(data, dict) else None
                if steps:
                    meta = {k: v for k, v in row.items() if k != "trajectory_data"}
                    out.append({"meta": meta, "steps": steps})
            except json.JSONDecodeError:
                continue
        return out
    # 标准：time + 关节列
    time_key = None
    for k in ("time", "simulation_time", "t", "timestamp"):
        if k in first:
            time_key = k
            break
    if time_key is None:
        raise ValueError("CSV 需包含 time/simulation_time/t 列")
    exclude = {time_key, "trajectory_id", "traj_id", "id"}
    joint_cols = [k for k in first.keys() if k not in exclude]
    if not joint_cols:
        raise ValueError("CSV 需包含关节列")

    def row_to_step(r):
        t = float(r.get(time_key, 0))
        joints = {}
        for j in joint_cols:
            try:
                joints[j] = float(r.get(j, 0))
            except (ValueError, TypeError):
                joints[j] = 0.0
        return {"simulation_time": t, "state": {"robot_joints": joints}}

    if "trajectory_id" in first or "traj_id" in first:
        tid_key = "trajectory_id" if "trajectory_id" in first else "traj_id"
        groups = {}
        for r in rows:
            tid = r.get(tid_key, "")
            if tid not in groups:
                groups[tid] = []
            groups[tid].append(row_to_step(r))
        out = []
        for tid in sorted(groups.keys(), key=lambda x: (str(x), x)):
            steps = groups[tid]
            steps.sort(key=lambda s: s["simulation_time"])
            out.append({"meta": {"trajectory_id": tid}, "steps": steps})
        return out
    steps = [row_to_step(r) for r in rows]
    steps.sort(key=lambda s: s["simulation_time"])
    return [{"meta": {}, "steps": steps}]


def steps_to_arrays(steps: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """
    将 steps 转为 (times, joint_data, joint_names)。
    joint_data[name] 为 1D 数组，与 times 等长。
    """
    if not steps:
        raise ValueError("steps 为空")
    joint_names = sorted(steps[0]["state"]["robot_joints"].keys())
    times = np.array([s["simulation_time"] for s in steps], dtype=float)
    joint_data = {
        name: np.array([s["state"]["robot_joints"].get(name, 0) for s in steps], dtype=float)
        for name in joint_names
    }
    return times, joint_data, joint_names


# --------------- 1. 平滑度（加速度、jerk） ---------------

def smoothness_metrics(
    times: np.ndarray,
    joint_data: dict[str, np.ndarray],
    joint_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    计算轨迹平滑度：离散加速度与 jerk（绝对值均值/最大值，按关节与整体）。

    参数:
        times: 时间序列 (N,)，单调递增。
        joint_data: 关节名 -> 关节角序列 (N,)。
        joint_names: 参与计算的关节名；None 表示 joint_data 的全部 key。

    返回:
        {
          "acceleration": {"mean_abs": float, "max_abs": float, "per_joint": {name: {"mean_abs", "max_abs"}}},
          "jerk":         {"mean_abs": float, "max_abs": float, "per_joint": {name: {"mean_abs", "max_abs"}}},
        }
    """
    if joint_names is None:
        joint_names = list(joint_data.keys())
    n = len(times)
    if n < 3:
        return {
            "acceleration": {"mean_abs": np.nan, "max_abs": np.nan, "per_joint": {}},
            "jerk": {"mean_abs": np.nan, "max_abs": np.nan, "per_joint": {}},
        }

    dt = np.diff(times)
    dt_safe = np.where(dt > 1e-10, dt, 1e-10)
    acc_per_joint = {}
    jerk_per_joint = {}
    all_acc = []
    all_jerk = []

    for name in joint_names:
        q = np.asarray(joint_data[name], dtype=float)
        if len(q) != n:
            continue
        # 速度 (中心差分，长度 n-2)
        dq = (q[1:] - q[:-1]) / dt_safe
        vel = (dq[1:] + dq[:-1]) * 0.5  # 在 i=1..n-2
        dt_mid = (times[2:] - times[:-2]) * 0.5
        dt_mid_safe = np.where(np.abs(dt_mid) > 1e-10, dt_mid, 1e-10)
        acc = (dq[1:] - dq[:-1]) / dt_mid_safe
        acc_abs = np.abs(acc)
        acc_per_joint[name] = {"mean_abs": float(np.mean(acc_abs)), "max_abs": float(np.max(acc_abs))}
        all_acc.extend(acc_abs.tolist())
        if len(acc) >= 2:
            dt_j = (times[3:] - times[1:-2]) * 0.5
            dt_j_safe = np.where(np.abs(dt_j) > 1e-10, dt_j, 1e-10)
            jerk = np.diff(acc) / dt_j_safe
            jerk_abs = np.abs(jerk)
            jerk_per_joint[name] = {"mean_abs": float(np.mean(jerk_abs)), "max_abs": float(np.max(jerk_abs))}
            all_jerk.extend(jerk_abs.tolist())
        else:
            jerk_per_joint[name] = {"mean_abs": 0.0, "max_abs": 0.0}

    return {
        "acceleration": {
            "mean_abs": float(np.mean(all_acc)) if all_acc else np.nan,
            "max_abs": float(np.max(all_acc)) if all_acc else np.nan,
            "per_joint": acc_per_joint,
        },
        "jerk": {
            "mean_abs": float(np.mean(all_jerk)) if all_jerk else np.nan,
            "max_abs": float(np.max(all_jerk)) if all_jerk else np.nan,
            "per_joint": jerk_per_joint,
        },
    }


# --------------- 2. 优化前后位置偏差（关节 / 末端） ---------------

def position_deviation_joint(
    times_before: np.ndarray,
    joint_data_before: dict[str, np.ndarray],
    times_after: np.ndarray,
    joint_data_after: dict[str, np.ndarray],
    joint_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    在「优化后」的时间网格上，将优化前轨迹插值到该网格，计算关节空间位置偏差。
    插值方式与 smooth_resampled_traj 一致：三次样条（CubicSpline, natural），便于与重采样结果可比。

    参数:
        times_before, joint_data_before: 优化前轨迹。
        times_after, joint_data_after: 优化后轨迹（作为参考时间网格）。
        joint_names: 参与计算的关节；None 表示两边 key 的交集。

    返回:
        {"max_abs_dev": float, "mean_abs_dev": float, "per_joint": {name: {"max_abs", "mean_abs"}}}
    """
    joint_names = joint_names or list(set(joint_data_before.keys()) & set(joint_data_after.keys()))
    if not joint_names or len(times_after) == 0:
        return {"max_abs_dev": np.nan, "mean_abs_dev": np.nan, "per_joint": {}}
    if len(times_before) < 2:
        return {"max_abs_dev": np.nan, "mean_abs_dev": np.nan, "per_joint": {}}

    dev_per_joint = {}
    all_dev = []
    for name in joint_names:
        if name not in joint_data_before or name not in joint_data_after:
            continue
        q_b = np.asarray(joint_data_before[name])
        q_a = np.asarray(joint_data_after[name])
        # 与 smooth_resampled_traj 一致：三次样条 natural
        cs_b = CubicSpline(times_before, q_b, bc_type="natural")
        q_b_interp = cs_b(times_after)
        dev = np.abs(q_a - q_b_interp)
        dev_per_joint[name] = {"max_abs": float(np.max(dev)), "mean_abs": float(np.mean(dev))}
        all_dev.extend(dev.tolist())

    if not all_dev:
        return {"max_abs_dev": np.nan, "mean_abs_dev": np.nan, "per_joint": {}}
    return {
        "max_abs_dev": float(np.max(all_dev)),
        "mean_abs_dev": float(np.mean(all_dev)),
        "per_joint": dev_per_joint,
    }


def position_deviation_ee(
    times_before: np.ndarray,
    joint_data_before: dict[str, np.ndarray],
    times_after: np.ndarray,
    joint_data_after: dict[str, np.ndarray],
    joint_names: list[str],
    arm_joint_names: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    在优化后时间网格上，将优化前末端位置插值到该网格，计算末端位置偏差（L2 范数，米）。
    需要 7 轴臂关节与 roboticstoolbox；不可用时返回 None。

    参数:
        times_before, joint_data_before: 优化前轨迹。
        times_after, joint_data_after: 优化后轨迹。
        joint_names: 完整关节名列表。
        arm_joint_names: 用于 FK 的 7 个臂关节名；None 时取 joint_names 中 franka/panda_joint1..7。

    返回:
        {"max_dev_m": float, "mean_dev_m": float, "per_time_step": np.ndarray} 或 None。
    """
    if not _RTB_AVAILABLE:
        return None
    arm_names = arm_joint_names or [n for n in joint_names if "panda_joint" in n and "finger" not in n]
    if len(arm_names) != 7:
        return None
    try:
        panda = rtb.models.DH.Panda()
    except Exception:
        return None
    for name in arm_names:
        if name not in joint_data_before or name not in joint_data_after:
            return None
    q_b = np.column_stack([joint_data_before[n] for n in arm_names])
    q_a = np.column_stack([joint_data_after[n] for n in arm_names])
    ee_b = np.array([np.asarray(panda.fkine(q_b[i]).t).ravel() for i in range(len(q_b))])
    ee_a = np.array([np.asarray(panda.fkine(q_a[i]).t).ravel() for i in range(len(q_a))])
    if ee_b.ndim == 1:
        ee_b = ee_b.reshape(1, -1)
    if ee_a.ndim == 1:
        ee_a = ee_a.reshape(1, -1)
    if len(times_after) == 0:
        return None
    # 与 smooth_resampled_traj 一致：三次样条插值（末端 x,y,z 各做一次）
    ee_b_interp = np.column_stack([
        CubicSpline(times_before, ee_b[:, d], bc_type="natural")(times_after)
        for d in range(ee_b.shape[1])
    ])
    dev = np.linalg.norm(ee_a - ee_b_interp, axis=1)
    return {
        "max_dev_m": float(np.max(dev)),
        "mean_dev_m": float(np.mean(dev)),
        "per_time_step": dev.tolist(),
    }


# --------------- 3. 移除帧比例（严格：去静止帧） ---------------

# 与 smooth_resampled_traj 默认一致
DEFAULT_MERGE_THRESHOLD = 1e-5


def count_frames_after_clean(steps: list[dict], merge_threshold: float = DEFAULT_MERGE_THRESHOLD) -> int:
    """
    对轨迹做与 smooth_resampled_traj.clean_trajectory 相同的静止帧合并，返回合并后的关键帧数。
    任一关节相对上一关键帧变化超过 merge_threshold 则保留该帧，否则视为静止并合并。

    参数:
        steps: 原始 steps（含 simulation_time、state.robot_joints）。
        merge_threshold: 判定静止的关节变化阈值。

    返回:
        去静止帧后的关键帧数量。
    """
    if not steps:
        return 0
    joint_names = sorted(steps[0]["state"]["robot_joints"].keys())
    filtered = [steps[0]]
    for i in range(1, len(steps)):
        curr = steps[i]
        prev = filtered[-1]
        is_static = True
        for name in joint_names:
            vc = curr["state"]["robot_joints"].get(name, 0)
            vp = prev["state"]["robot_joints"].get(name, 0)
            if abs(vc - vp) > merge_threshold:
                is_static = False
                break
        if not is_static:
            filtered.append(curr)
    return len(filtered)


def removed_frames_ratio(
    num_original_steps: int,
    num_after_clean_steps: int,
) -> float:
    """
    移除帧数占原始总帧数的比例（严格表示去静止帧的压缩程度）。

    参数:
        num_original_steps: 原始轨迹总帧数。
        num_after_clean_steps: 去静止帧后的关键帧数（不含重采样）。

    返回:
        (num_original_steps - num_after_clean_steps) / num_original_steps；
        num_original_steps <= 0 时返回 0.0。
    """
    if num_original_steps <= 0:
        return 0.0
    removed = max(0, num_original_steps - num_after_clean_steps)
    return removed / num_original_steps


# --------------- 汇总：单条轨迹的完整 metrics ---------------

def compute_trajectory_metrics(
    steps_before: list[dict],
    steps_after: list[dict],
    num_original_steps: int | None = None,
    num_after_clean_steps: int | None = None,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    include_ee: bool = True,
) -> dict[str, Any]:
    """
    对单条轨迹计算：平滑度（加速度、jerk）、关节/末端位置偏差、移除帧比例。
    使用优化后轨迹作为时间网格与平滑度计算对象；优化前用于插值比较。
    removed_frames_ratio 严格表示去静止帧：用与 smooth_resampled_traj 相同的合并逻辑计算去静止帧后帧数。

    参数:
        steps_before: 优化前 steps。
        steps_after: 优化后 steps。
        num_original_steps: 原始总帧数；None 时用 len(steps_before)。
        num_after_clean_steps: 去静止帧后的关键帧数；None 时用 count_frames_after_clean(steps_before, merge_threshold) 严格计算。
        merge_threshold: 去静止帧判定阈值（与 smooth_resampled_traj 一致）；仅当 num_after_clean_steps 为 None 时使用。
        include_ee: 是否计算末端偏差。

    返回:
        {"smoothness_before", "smoothness_after", "position_deviation_joint", "position_deviation_ee"?, "removed_frames_ratio"}
    """
    t_b, j_b, jnames = steps_to_arrays(steps_before)
    t_a, j_a, _ = steps_to_arrays(steps_after)
    joint_names = list(j_b.keys())

    metrics = {
        "smoothness_before": smoothness_metrics(t_b, j_b, joint_names),
        "smoothness_after": smoothness_metrics(t_a, j_a, joint_names),
        "position_deviation_joint": position_deviation_joint(t_b, j_b, t_a, j_a, joint_names),
    }
    if include_ee:
        ee_dev = position_deviation_ee(t_b, j_b, t_a, j_a, joint_names)
        if ee_dev is not None:
            metrics["position_deviation_ee"] = ee_dev

    num_orig = num_original_steps if num_original_steps is not None else len(steps_before)
    if num_after_clean_steps is not None:
        num_clean = num_after_clean_steps
    else:
        num_clean = count_frames_after_clean(steps_before, merge_threshold)
    metrics["removed_frames_ratio"] = removed_frames_ratio(num_orig, num_clean)
    return metrics


# --------------- 保存为文件 ---------------

def _flatten_metrics_one(m: dict) -> dict:
    """将单条轨迹的 metrics 压平为一层 dict，便于写 CSV 一行。"""
    row = {"trajectory_index": m.get("trajectory_index", -1)}
    if m.get("source_before") is not None:
        row["source_before"] = m["source_before"]
    if m.get("source_after") is not None:
        row["source_after"] = m["source_after"]
    if m.get("file_trajectory_index") is not None:
        row["file_trajectory_index"] = m["file_trajectory_index"]
    sb = m.get("smoothness_before", {})
    sa = m.get("smoothness_after", {})
    for name, d in [("smoothness_before", sb), ("smoothness_after", sa)]:
        acc = d.get("acceleration", {})
        jerk = d.get("jerk", {})
        row[f"{name}_acc_mean_abs"] = acc.get("mean_abs")
        row[f"{name}_acc_max_abs"] = acc.get("max_abs")
        row[f"{name}_jerk_mean_abs"] = jerk.get("mean_abs")
        row[f"{name}_jerk_max_abs"] = jerk.get("max_abs")
    pdj = m.get("position_deviation_joint", {})
    row["position_deviation_joint_max_abs"] = pdj.get("max_abs_dev")
    row["position_deviation_joint_mean_abs"] = pdj.get("mean_abs_dev")
    pde = m.get("position_deviation_ee")
    if pde:
        row["position_deviation_ee_max_m"] = pde.get("max_dev_m")
        row["position_deviation_ee_mean_m"] = pde.get("mean_dev_m")
    row["removed_frames_ratio"] = m.get("removed_frames_ratio")
    meta = m.get("meta_before", {})
    if isinstance(meta, dict):
        for k in ("attempt_id", "task_id", "user_id", "id"):
            if k in meta and not isinstance(meta[k], (dict, list)):
                row[f"meta_{k}"] = meta[k]
    return row


def save_metrics_json(data: dict, path: str) -> None:
    """将完整 metrics 保存为 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_metrics_csv(metrics_per_trajectory: list, path: str) -> None:
    """
    将每条轨迹的 metrics 压平后保存为 CSV（一行一条轨迹）。
    列：trajectory_index, smoothness_before/after 汇总, position_deviation_*, removed_frames_ratio, meta_* 等。
    """
    if not metrics_per_trajectory:
        return
    import csv
    rows = [_flatten_metrics_one(m) for m in metrics_per_trajectory]
    fieldnames = sorted(set().union(*(set(r.keys()) for r in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


# --------------- Main ---------------

def parse_args():
    p = argparse.ArgumentParser(description="轨迹优化指标：平滑度、位置偏差、移除帧比例。支持单文件或文件夹批量。")
    p.add_argument("--before", "-b", default=None, help="优化前轨迹文件或目录（.json/.csv；目录时与 --after 目录内文件按 stem 配对）")
    p.add_argument("--after", "-a", default=None, help="优化后轨迹文件或目录（同上；配对规则：after 的 stem 为 before 的 stem 或 stem_smoothed）")
    p.add_argument("--output", "-o", default=None, help="输出文件路径（根据 --output-format 写 JSON 和/或 CSV）")
    p.add_argument("--output-format", choices=["json", "csv", "both"], default="json",
                   help="输出格式：json=仅 JSON；csv=仅 CSV（扁平化，一行一条轨迹）；both=同时写 .json 与 .csv")
    p.add_argument("--no-ee", action="store_true", help="不计算末端位置偏差")
    p.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD,
                   help=f"去静止帧判定阈值（与 smooth_resampled_traj 一致），默认 {DEFAULT_MERGE_THRESHOLD}")
    p.add_argument("--test", action="store_true", help="运行内置测试")
    args = p.parse_args()
    if not args.test and (args.before is None or args.after is None):
        p.error("未指定 --test 时需同时提供 --before 与 --after")
    return args


def _pair_before_after_dirs(before_dir: Path, after_dir: Path) -> list[tuple[Path, Path]]:
    """目录下 .json/.csv 按 stem 配对：before 的 stem 对应 after 的 stem 或 stem_smoothed。"""
    before_files = sorted(
        set(before_dir.glob("*.json")) | set(before_dir.glob("*.csv")),
        key=lambda p: p.name,
    )
    after_by_stem: dict[str, Path] = {}
    for p in set(after_dir.glob("*.json")) | set(after_dir.glob("*.csv")):
        stem = p.stem
        norm = stem.replace("_smoothed", "") if stem.endswith("_smoothed") else stem
        after_by_stem[norm] = p
    pairs = []
    for bf in before_files:
        af = after_by_stem.get(bf.stem)
        if af is None and bf.stem.endswith("_smoothed"):
            af = after_by_stem.get(bf.stem.replace("_smoothed", ""))
        if af is not None:
            pairs.append((bf, af))
        else:
            print(f"[Warn] 未找到配对，跳过: {bf.name}")
    return pairs


def main():
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)

    if before_path.is_dir() and after_path.is_dir():
        # 批量：目录对目录
        pairs = _pair_before_after_dirs(before_path, after_path)
        if not pairs:
            print("[Error] 未找到任何可配对的 before/after 文件")
            return
        print(f"[Batch] 共 {len(pairs)} 对文件")
        results = []
        global_idx = 0
        for bf, af in pairs:
            before_list = load_trajectories(str(bf))
            after_list = load_trajectories(str(af))
            n = min(len(before_list), len(after_list))
            if len(before_list) != len(after_list):
                print(f"[Warn] {bf.name} / {af.name} 轨迹条数不一致，按前 min 条评估")
            for i in range(n):
                m = compute_trajectory_metrics(
                    before_list[i]["steps"],
                    after_list[i]["steps"],
                    merge_threshold=args.merge_threshold,
                    include_ee=not args.no_ee,
                )
                m["trajectory_index"] = global_idx
                m["source_before"] = bf.name
                m["source_after"] = af.name
                m["file_trajectory_index"] = i
                if before_list[i].get("meta"):
                    meta = before_list[i]["meta"]
                    m["meta_before"] = {k: v for k, v in meta.items() if k != "trajectory_data"}
                results.append(m)
                acc_b = m["smoothness_before"]["acceleration"]["max_abs"]
                acc_a = m["smoothness_after"]["acceleration"]["max_abs"]
                print(f"[{bf.name} Traj {i}] acc_max before={acc_b:.6f} after={acc_a:.6f}, "
                      f"joint_dev(max)={m['position_deviation_joint']['max_abs_dev']:.6f}, "
                      f"removed_ratio={m['removed_frames_ratio']:.4f}")
                global_idx += 1
        n = len(results)
    else:
        if before_path.is_dir() or after_path.is_dir():
            print("[Error] --before 与 --after 须同时为文件或同时为目录")
            return
        before_list = load_trajectories(args.before)
        after_list = load_trajectories(args.after)
        if len(before_list) != len(after_list):
            print(f"[Warn] 前后轨迹条数不一致: before={len(before_list)}, after={len(after_list)}，按索引对齐前 min 条")
        n = min(len(before_list), len(after_list))
        results = []
        for i in range(n):
            m = compute_trajectory_metrics(
                before_list[i]["steps"],
                after_list[i]["steps"],
                merge_threshold=args.merge_threshold,
                include_ee=not args.no_ee,
            )
            m["trajectory_index"] = i
            if before_list[i].get("meta"):
                meta = before_list[i]["meta"]
                m["meta_before"] = {k: v for k, v in meta.items() if k != "trajectory_data"}
            results.append(m)
            acc_b = m["smoothness_before"]["acceleration"]["max_abs"]
            acc_a = m["smoothness_after"]["acceleration"]["max_abs"]
            print(f"[Traj {i}] smoothness: acc_max before={acc_b:.6f} after={acc_a:.6f}, "
                  f"joint_dev(max)={m['position_deviation_joint']['max_abs_dev']:.6f}, "
                  f"removed_ratio={m['removed_frames_ratio']:.4f}")

    out = {"num_trajectories": n, "metrics_per_trajectory": results}
    if args.output:
        base = Path(args.output)
        stem = base.stem
        parent = base.parent
        fmt = getattr(args, "output_format", "json")
        if fmt == "json":
            path = parent / (stem + ".json") if base.suffix.lower() != ".json" else base
            save_metrics_json(out, str(path))
            print(f"[Saved] {path}")
        elif fmt == "csv":
            path = parent / (stem + ".csv") if base.suffix.lower() != ".csv" else base
            save_metrics_csv(results, str(path))
            print(f"[Saved] {path}")
        else:
            path_json = parent / (stem + ".json")
            path_csv = parent / (stem + ".csv")
            save_metrics_json(out, str(path_json))
            save_metrics_csv(results, str(path_csv))
            print(f"[Saved] {path_json}")
            print(f"[Saved] {path_csv}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))




if __name__ == "__main__":
    main()
