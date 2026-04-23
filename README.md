# MI PsychoPy 实验脚本

运动想象（MI）EEG 实验的刺激呈现程序。基于 PsychoPy，配合 OpenBCI + LSL 生态完成数据采集。

## 整体流程

```text
┌─────────────┐     UDP      ┌──────────────┐     LSL      ┌──────────────┐
│  PsychoPy   │──marker──→   │ OpenBCI GUI  │──EEG+Marker─→│ LabRecorder  │
│  (本脚本)    │              │  (硬件驱动)   │              │  (录制成XDF) │
└─────────────┘              └──────────────┘              └──────────────┘
```

- **PsychoPy（本脚本）**：呈现视觉刺激，在关键时间点通过 UDP 发送 marker（事件标记）
- **OpenBCI GUI**：驱动 Cyton 板采集 EEG，接收 marker 并与 EEG 数据一起通过 LSL 发布
- **LabRecorderCLI**：自动录制 LSL 流，保存为 XDF 文件（包含 EEG + marker + 元数据）

脚本会自动启动/停止 LabRecorderCLI，全程无需手动操作录制软件。

## 快速开始

### 1. 安装依赖

```bash
pip install psychopy pyyaml
```

> PsychoPy 需要 OpenGL 2.1+ 显卡支持。如安装遇阻，参考 [PsychoPy 官方安装指南](https://www.psychopy.org/download.html)。

### 2. 准备硬件与软件

运行脚本前，确保以下条件满足：

| 步骤 | 操作 | 说明 |
|------|------|------|
| ① | 启动 **OpenBCI GUI**（发布版，非源码） | 确保使用官方发布版 GUI |
| ② | 连接 **Cyton** 板，确认 EEG 波形正常 | 8 通道：C3, Cz, C4, P3, Pz, P4, O1, O2 |
| ③ | GUI **Marker Widget** → 开启 UDP 接收 | 默认地址 `127.0.0.1:12350` |
| ④ | GUI **Networking Widget** → 开启 LSL 输出 | 勾选 TimeSeriesRaw + Marker 两项 |

> ⚠️ OpenBCI GUI 的 Marker UDP 和 LSL 输出配置**不会自动持久化**，每次重启 GUI 都需要重新勾选。只需点几下即可完成。

### 3. 首次试采

```bash
python run_mi_experiment.py
```

直接运行会弹出配置对话框，在对话框中选择范式、被试信息等，点击 OK 即可开始。首次建议用 `pure_mi`（传统运动想象）验证整个链路是否通畅。

如果需要跳过对话框，用命令行参数：

```bash
python run_mi_experiment.py --trial-mode pure_mi
```

### 4. 确认数据采集正常

实验结束后检查以下内容：

- `logs/` 目录下生成了 `*_plan.json`（trial 方案）和 `*_events.csv`（事件日志）
- OpenBCI GUI 的 Marker Widget 有收到 marker 数值
- XDF 文件保存到 `config_default.yaml` 中 `labrecorder_study_root` 指定的路径

## 实验范式

脚本支持 7 种实验范式，通过对话框下拉框或 `--trial-mode` 参数选择：

| 范式 | trial-mode | 说明 | Trial 结构 |
|------|-----------|------|-----------|
| 传统 MI | `pure_mi` | 基础运动想象 | fixation → cue → mi → iti |
| AO+MI | `ao_mi` | 动作观察+运动想象 | fixation → cue → ao_prime → ao_mi → mi_only → iti |
| MI+SSVEP | `mi_ssvep` | SSVEP 闪烁+运动想象 | fixation → cue → mi_ssvep → iti |
| MI+P300 | `mi_p300` | P300 oddball+运动想象 | fixation → cue → mi_p300 → iti |
| 纯箭头 MI | `mi_arrow` | 仅箭头提示的 MI baseline | fixation → arrow_cue → arrow_mi → iti |
| SSVEP 觉醒增强 | `mi_ssvep_arousal` | 中央 SSVEP 提高觉醒度+MI | fixation → arousal_cue → arousal_task → iti |
| 串行 SSVEP→MI | `mi_ssvep_serial` | 先 SSVEP cue 再纯 MI | fixation → serial_ssvep_cue → serial_gap → serial_mi → iti |

**分类模式**：
- `binary`：左手 / 右手（所有范式均支持）
- `ternary`：左手 / 右手 / 被动静息（仅 `pure_mi`）

### 新范式设计要点

**纯箭头 MI (`mi_arrow`)**：无图片、无 SSVEP，仅中央箭头提示方向。作为 MI 的 ground truth baseline，用于对比 SSVEP/P300 混合范式是否引入额外信号泄露。

**SSVEP 觉醒增强 (`mi_ssvep_arousal`)**：中央单点 SSVEP 闪烁，频率**不编码**左右条件（所有 trial 使用相同频率）。支持固定频率和 per-trial 随机频率两种模式。目的：用 SSVEP 提高被试觉醒度，但不让频率泄露类别标签。

**串行 SSVEP→MI (`mi_ssvep_serial`)**：cue 段 SSVEP 标记方向 → 间隔段无 SSVEP → 纯 MI 段。参考清华方案，通过时序分离避免 SSVEP 与 MI 的信号混叠。

## 配置

配置优先级：**对话框 > CLI 参数 > 用户 YAML > 内置默认 YAML**

### 对话框（主要配置入口）

运行脚本后自动弹出 tkinter 对话框，包含：

- **通用设置**：被试编号、session、run、范式选择、全屏、显示器选择
- **范式专属面板**：根据选择的 trial-mode 自动切换显示
  - MI+SSVEP：闪烁模式、波形、频率、显示模式
  - MI+P300：闪烁模式、目标概率
  - SSVEP Arousal：频率模式（fixed/random）、波形、cue/task 样式
  - SSVEP Serial：cue 模式（频率编码/同频）、各段时长、波形、样式

> 对话框中的条件参数会根据选择自动展开/隐藏。例如频率模式选 "fixed" 时只显示固定频率输入框，选 "random" 时才显示频率范围。

### CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--participant` | P001 | 被试编号 |
| `--session` | S001 | Session 编号 |
| `--run` | 1 | Run 编号 |
| `--mode` | pilot | pilot=快速验证 / main=正式采集 |
| `--class-mode` | binary | binary / ternary |
| `--trial-mode` | pure_mi | 实验范式 |
| `--blocks` | 自动 | 覆盖 block 数（pilot=2, main=4） |
| `--fullscreen` | off | 全屏模式 |
| `--display-index` | 0 | 显示器编号 |
| `--config` | config_default.yaml | YAML 配置文件路径 |

### YAML 配置文件

所有时序、频率、布局参数在 YAML 中按范式分区配置。默认配置：[`config_default.yaml`](config_default.yaml)（每项均有中文注释）

自定义方式：

```bash
cp config_default.yaml my_experiment.yaml
# 编辑 my_experiment.yaml
python run_mi_experiment.py --config my_experiment.yaml
```

YAML 分区：

| 分区 | 说明 |
|------|------|
| `general` | 时序、窗口、网络、LabRecorder、图片缩放 |
| `pure_mi` | MI 时长、cue/任务图片路径 |
| `ao_mi` | AO 时序、视频文件 |
| `mi_ssvep` | SSVEP 频率、闪烁模式、显示布局 |
| `mi_p300` | P300 SOA、闪光时长、闪烁模式 |
| `mi_arrow` | 箭头样式、颜色、大小 |
| `mi_ssvep_arousal` | 频率模式、波形、cue/task 样式 |
| `mi_ssvep_serial` | cue 频率/模式、间隔时长、MI 时长 |

## 示例命令

```bash
# 传统 MI（首次推荐）
python run_mi_experiment.py --trial-mode pure_mi

# 纯箭头 MI baseline
python run_mi_experiment.py --trial-mode mi_arrow

# MI+SSVEP
python run_mi_experiment.py --trial-mode mi_ssvep

# SSVEP 觉醒增强 MI
python run_mi_experiment.py --trial-mode mi_ssvep_arousal

# 串行 SSVEP→MI
python run_mi_experiment.py --trial-mode mi_ssvep_serial

# MI+P300
python run_mi_experiment.py --trial-mode mi_p300

# 三分类传统 MI
python run_mi_experiment.py --trial-mode pure_mi --class-mode ternary

# 正式采集（全屏、副屏显示）
python run_mi_experiment.py --mode main --trial-mode pure_mi \
  --participant P001 --session S001 --run 1 --fullscreen --display-index 1

# 使用自定义配置
python run_mi_experiment.py --config my_config.yaml --trial-mode mi_ssvep_arousal
```

## 运行时按键

| 按键 | 功能 |
|------|------|
| SPACE | 开始 / 继续 |
| ESC | 安全中断 session |
| F11 | 切换全屏 / 窗口 |
| 1 / 2 / 3 | 切换预设窗口尺寸 |

> ESC 中断时 XDF 文件会正常保存，不会丢失数据。

## 输出文件

每次 run 自动创建独立目录：

```text
logs/P001_S001_run-01_pilot_20260422-220335/
├── *_plan.json      # trial 顺序、marker 编码、配置快照
└── *_events.csv     # 事件日志（block、trial、condition、marker、时间戳）
```

XDF 文件保存在 `config_default.yaml` 中 `labrecorder_study_root` 指定的路径下，按被试/session/类别/task 自动归档。

## Marker 编码

定义在 [`markers.py`](markers.py)：

| 名称 | 值 | 触发时机 |
|------|---:|---------|
| `fixation_on` | 5 | fixation 开始 |
| `cue_left` / `cue_right` / `cue_rest` | 11 / 12 / 13 | cue 开始 |
| `mi_left` / `mi_right` / `mi_rest` | 21 / 22 / 23 | pure MI 开始 |
| `task_off` | 29 | 任务相位结束 |
| `ao_prime_left` / `ao_prime_right` | 31 / 32 | AO prime 开始 |
| `ao_mi_left` / `ao_mi_right` | 41 / 42 | AO+MI overlap 开始 |
| `mi_only_left` / `mi_only_right` | 51 / 52 | MI-only tail 开始 |
| `ssvep_left` / `ssvep_right` | 61 / 62 | MI+SSVEP 开始 |
| `ssvep_gaze_left` / `ssvep_gaze_right` | 71 / 72 | gaze shift 标记 |
| `p300_target_flash` / `p300_nontarget_flash` | 81 / 82 | P300 闪光事件 |
| `arrow_cue_left` / `arrow_cue_right` | 101 / 102 | 箭头 cue 开始 |
| `arrow_mi_left` / `arrow_mi_right` | 111 / 112 | 箭头 MI 开始 |
| `arousal_cue_left` / `arousal_cue_right` | 121 / 122 | SSVEP Arousal cue 开始 |
| `arousal_task_left` / `arousal_task_right` | 131 / 132 | SSVEP Arousal task 开始 |
| `serial_ssvep_cue_left` / `serial_ssvep_cue_right` | 141 / 142 | 串行 SSVEP cue 开始 |
| `serial_gap` | 145 | 串行间隔开始 |
| `serial_mi_left` / `serial_mi_right` | 151 / 152 | 串行 MI 开始 |
| `audio_fb_cue_left` / `audio_fb_cue_right` | 161 / 162 | 听觉反馈 cue（预留） |
| `audio_fb_task_left` / `audio_fb_task_right` | 171 / 172 | 听觉反馈 task（预留） |
| `block_start` / `block_end` | 90 / 91 | block 边界 |
| `session_end` | 99 | session 结束 |

## Trial 数量

| 范式 | pilot (2 blocks) | main (4 blocks) |
|------|:---:|:---:|
| pure_mi + ternary | 24 | 48 |
| pure_mi + binary | 16 | 32 |
| ao_mi / mi_ssvep / mi_p300 / mi_arrow / mi_ssvep_arousal / mi_ssvep_serial | 16 | 32 |

> 默认 `repeats_per_class: 10`（每 block 每类 10 次）。可在 YAML 或对话框中调整。

## EEG 通道布局

当前使用 8 通道：**C3, Cz, C4, P3, Pz, P4, O1, O2**

| 通道 | 用途 |
|------|------|
| C3 / Cz / C4 | 运动想象主相关通道 |
| P3 / Pz / P4 | 感觉运动与后部区域过渡带 |
| O1 / O2 | SSVEP 主相关通道 |

## 常见问题

**Q: 启动后 OpenBCI GUI 收不到 marker？**
检查 GUI Marker Widget 是否开启了 UDP 接收，地址和端口是否与 `config_default.yaml` 中 `udp_ip` / `udp_port` 一致。

**Q: LabRecorder 没有自动录制？**
检查 GUI Networking Widget 是否开启了 LSL 输出（TimeSeriesRaw + Marker）。脚本会自动检测 LSL 流是否可用，并在起始屏显示状态。

**Q: SSVEP 闪烁看起来不平滑？**
确保显示器刷新率设置正确（默认 60Hz）。在 YAML 中修改 `refresh_rate` 为实际刷新率。高频率 SSVEP（>20Hz）建议使用 sine 波形。

**Q: 想在副屏全屏运行？**
对话框中选择第二块显示器，或使用 `--fullscreen --display-index 1`（0=主屏，1=副屏）。

**Q: 手动录制（不自动启动 LabRecorderCLI）？**
在 YAML 中设置 `labrecorder_auto_record: false`，然后手动启动 LabRecorder GUI。
