# smooth_resampled_traj 使用说明
# 注意，这里主要是smooth_resampled_traj.py的使用说明
---

## Requirement

- Python 3.10+
- 依赖：`numpy`、`scipy`、`matplotlib`
- **可选**：`roboticstoolbox`（用于末端位置 3D 对比图；未安装时自动退化为关节空间对比图）

安装依赖：

```bash
pip install numpy scipy matplotlib
# 可选：末端位置图需要
pip install roboticstoolbox
```

---

## 基本用法

```bash
python smooth_resampled_traj.py --input <输入文件> --output <输出文件> [选项]
```

**必选参数：**

| 参数 | 简写 | 说明 |
|------|------|------|
| `--input` | `-i` | 输入文件路径（`.json` 或 `.csv`） |
| `--output` | `-o` | 输出文件路径（`.json` 或 `.csv`） |

**常用可选参数：**

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `-m` | `smooth_resample` | 模式：`resample`（仅重采样）或 `smooth_resample`（平滑+重采样） |
| `--dt` | | `0.015` | 目标时间间隔（秒），即重采样后的步长 |
| `--merge-threshold` | | `1e-5` | 判定“静止帧”的关节变化阈值，小于该值的连续帧会被合并 |
| `--smooth-window` | | `15` | Savitzky-Golay 平滑窗口长度（奇数，仅 `smooth_resample` 时有效） |
| `--smooth-poly` | | `3` | Savitzky-Golay 多项式阶数 |
| `--format` | `-f` | `auto` | 输入格式：`auto`（按扩展名）、`json`、`csv` |
| `--no-visualize` | | | 不生成可视化图片 |
| `--visualize-path` | | 根据输出名自动 | 可视化图片保存路径 |

---

## Example

**1. resample only（dt = 0.015）：**

```bash
python smooth_resampled_traj.py -i trajectory.json -o trajectory_resampled.json --mode resample --dt 0.015
```

**2. resample + smooth (dt = 0.015)：**

```bash
python smooth_resampled_traj.py -i trajectory.json -o trajectory_smoothed_resampled.json --mode smooth_resample --dt 0.015
```

**3. format：**

```bash
python smooth_resampled_traj.py -i trajectory_sample.csv -o out.json --mode resample --dt 0.02 --no-visualize
```

**4. example for other parameters**

```bash
python smooth_resampled_traj.py -i trajectory.json -o out.json --mode smooth_resample \
  --dt 0.02 --merge-threshold 1e-6 --smooth-window 21 --smooth-poly 3
```

**5. output CSV：**

```bash
python smooth_resampled_traj.py -i trajectory.json -o trajectory_out.csv --mode resample --dt 0.015
```

---

## 输入格式说明

### JSON

- **单条轨迹**：根对象包含 `trajectory_data.steps`，例如：
  ```json
  {
    "attempt_id": 26412,
    "trajectory_data": {
      "steps": [
        { "simulation_time": 0.025, "state": { "robot_joints": { "joint1": 0.1, ... } } },
        ...
      ]
    }
  }
  ```
- **多条轨迹**：根对象包含 `trajectories` 数组，每项为上述结构；或根节点直接为数组 `[{ ... }, { ... }]`。

### CSV

- **表头**：必须包含**时间列**（`time`、`simulation_time` 或 `t` 之一）以及若干**关节列**（列名即关节名）。
- **单条轨迹**：整表为一组 `(time, joint1, joint2, ...)`。
- **多条轨迹**：增加 `trajectory_id`（或 `traj_id`）列，按该列分组为多条轨迹。

示例（单条轨迹）：

```csv
time,franka/panda_joint1,franka/panda_joint2,...,franka/panda_finger_joint1,franka/panda_finger_joint2
0.025,0.0,-0.785,...,0.04,0.04
0.100,0.0,-0.784,...,0.04,0.04
```

