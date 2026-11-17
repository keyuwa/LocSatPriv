# -

# 卫星通信位置隐私实验

实验比较区域下行调度、精确位置报告和差分隐私位置报告。终端和业务请求均为合成数据，可使用合成波束或 RECORD 的历史轨道与公开波束模型。

环境使用 Python 3.12。

```sh
python -m pip install -r requirements.txt
python experiments/location_privacy/joint_experiment.py
```

使用 RECORD 历史轨道时，从作者仓库获取独立依赖。

```sh
git clone --branch usenix24 https://github.com/ErJedermann/RECORD.git experiments/control_state/RECORD
git -C experiments/control_state/RECORD checkout b661f362375a01ff6778212e74ecb11e32b8e746
git -C experiments/control_state/RECORD apply ../../../record-numpy-compat.patch
python experiments/location_privacy/joint_experiment.py --record-orbit
```

中文图使用系统提供的 Hiragino Sans GB、Noto Sans CJK SC 或 SimHei。运行参数、指标及实验边界见 [联合实验说明](experiments/location_privacy/联合实验.md)。已运行的历史轨道实验附有 [效果图](experiments/location_privacy/record-orbit-results/joint-effects.png) 和 [统计结果](experiments/location_privacy/record-orbit-results/summary.json)。

历史轨道的六组配对对照：

```sh
python experiments/location_privacy/joint_experiment.py --record-orbit --paired
```

使用 RECORD 原始定位仿真对比精确报告和差分报告。在应用前述 NumPy 兼容补丁后运行：

```sh
git -C experiments/control_state/RECORD apply ../../../record-extension/record-dp.patch
cp record-extension/record_dp_experiment.py experiments/control_state/RECORD/
(cd experiments/control_state/RECORD && python record_dp_experiment.py)
```
