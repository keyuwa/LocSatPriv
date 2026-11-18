# 卫星互联网位置隐私实验

比较波束观测、位置报告和联合观测下的定位隐私。坐标及距离以千米计，时间以秒计。

## 研究报告中的三个实验

| 实验 | 实现 | 已完成的结果 |
|---|---|---|
| 固定区域大小与下行定位误差 | [历史窗口回放](historical_windows.py) | [区域大小数据](historical-window-results/region-size.csv) · [对比图](historical-window-results/size-windows.png) |
| 加入差分隐私前后的平均定位误差与送达率 | [标签机制](schedule_privacy.py)与同一历史窗口回放 | [逐窗口结果](historical-window-results/summary.json) · [两组对比图](historical-window-results/dp-windows.png) |
| 多终端秘密共享与明文合批 | [秘密共享实验](secret_share_experiment.py) | [结果表](secret-share-results/section.md) · [逐批统计](secret-share-results/batches.csv) |

输入为 2022 年 2 月 14 日 Iridium 历史星历和 RECORD 公开带噪波束模型，三个窗口从 UTC 06:00、12:00、18:00 开始。终端与消息由仿真生成。前两个实验共用一次轨道回放；第三个实验使用两个逻辑服务器模拟加法秘密共享。

安装下节依赖后，从仓库根目录运行：

```sh
python historical_windows.py
python plot_historical_windows.py
python secret_share_experiment.py
python plot_secret_share.py
```

最后一条命令生成秘密共享实验的 Markdown 表格。中文图表需要系统安装 Noto Sans CJK SC、SimHei 或 macOS 中文字体。

## 安装

在项目根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 实验入口

| 文件 | 用途 | 默认输出 |
| --- | --- | --- |
| `demo.py` | 单次合成波束覆盖与候选位置收缩 | `result.svg` |
| `compare_dp.py` | 已知区域内重复差分隐私报告比较 | `compare-dp.json` |
| `report_privacy.py` | 位置报告隐私、效用和置信区间 | `reports.csv`、`report-summary.json`、`report-privacy.png` |
| `joint_experiment.py` | 波束与报告联合观测实验 | `joint-results/` |
| `record_orbit.py` | 本地 RECORD 星历和波束适配器 | 由联合实验调用 |

```sh
python demo.py
python compare_dp.py
python report_privacy.py
python joint_experiment.py --help
python joint_experiment.py --repeats 10 --users 100
python joint_experiment.py --record-orbit --output record-orbit-results
python joint_experiment.py --record-orbit --paired --output paired-orbit-results
```

重复运行会覆盖对应输出；联合实验可用 `--output` 保存到独立目录。`--fixed-total-budget` 用于固定总隐私预算的比较。

## 目录与资料

- `RECORD/`：配套模型、历史星历、波束数据和原始实验代码；其说明见 `RECORD/README.md`。
- `实验说明.md`：实验背景与方法说明。
- `联合实验.md`：联合实验设计与分析。
- `*-results/` 和根目录的 JSON、CSV、图片：已有实验产物，保留用于结果对照。

轨道实验直接读取本项目 `RECORD/beam_model/` 下的数据，无需相邻的 `control_state` 项目。历史星历用于实验复现，不代表当前卫星状态。

## 兼容历史布局

远端原有的 `experiments/location_privacy/` 与 `record-extension/` 保留用于历史复现，其安装和运行方式见 [历史说明](README-legacy.md)。日常实验优先使用本页所列根目录入口。
