# B601-DM + D405：抓放与电梯交互

当前工作版本的精简源码发布包，面向 B601-DM、RealSense D405、YOLOE 和 RapidOCR Torch CUDA。
保留现有可调用入口和控制算法；不重新命名日期脚本，不修改 SDK/URDF，使用独立发布副本。

## 入口

| 程序 | 用途 |
|---|---|
| `scripts/261113_my_elevator_journey.py` | 推荐总流程：抓放 → 外呼 → 轿内楼层按键 → LCD 到站识别 |
| `scripts/261110_my_grasp_then_elevator.py --floor 3` | 抓放后自动选择指定楼层 |
| `scripts/261105_my_main.py` | 原夹取/电梯模式选择入口 |
| `scripts/261112_my_call_panel.py` | 独立上下行两键识别 |
| `scripts/261113_my_floor_display.py` | 独立 LCD 楼层识别 |

其他日期脚本保留供独立视觉调试与现有回归测试调用；`scripts/main.py`、`object_detection.py` 为上游基线。

## 首先做无设备检查

在原来已验证的 `rebotarm` Conda 环境中运行：

```bash
conda activate rebotarm
python scripts/261113_my_elevator_journey.py --simulate
python -m unittest discover -s tests
```

以上两条命令不打开相机、不连接机械臂。测试包含照片、真实 URDF 的离线求解和模拟控制接口。
本发布版未重装任何依赖；环境版本记录见 [environment_reference.json](resources/environment_reference.json)。
这是版本参考而非完整安装锁文件，尚未验证全新环境安装。尤其不要照原环境混装多个 `cv2` 发行包。
上游安装说明保留于 [docs/upstream](docs/upstream)，其中的依赖配置不是当前 CUDA OCR 环境的安装方案。

## 恢复运行资源

Git 仓库不包含模型权重、文本编码器或本机标定数据。
完整路径、大小、SHA256 见 [runtime_assets.json](resources/runtime_assets.json)。
使用同一套硬件时，从原来的已验证工作目录按原相对路径复制这些文件：

- `models/yoloe-26l-seg.pt`、`mobileclip2_b.ts`
- `models/261104_my_rapidocr_torch/PP-OCRv6_rec_small.pth` 和 `ppocrv6_dict.txt`
- `config/calibration/realsense_d405/intrinsics.npz` 和 `hand_eye.npz`

换硬件或相机安装位置后必须重新取得匹配的标定，不应直接套用其他设备的手眼矩阵。
OCR 官方来源和下载计划由 `elevator/ocr.py:model_download_plan` 记录；本次发布整理没有自动下载模型或安装/升级软件；运行前请先补齐资源。
可运行下面的只读检查核对已经复制的资源：

```bash
python scripts/check_runtime_assets.py
```

当前 B601 控制 SDK 的运行源码、配置、DM/RS URDF 与 Mesh 已随仓库保留。
它包含本机已验证的兼容性修改，不要直接用上游最新版本覆盖。来源和逐文件哈希见 [source_provenance.json](resources/source_provenance.json)。
可选 GraspNet 不是当前电梯/普通夹取流程的依赖，未随精简版打包。

## 使用相机或机械臂

补齐资源后，先进行仅相机预览：

```bash
python scripts/261113_my_elevator_journey.py --dry-run --device 0 --ocr-device cuda
```

明确要执行真实动作时，才运行：

```bash
python scripts/261113_my_elevator_journey.py --execute --device 0 --ocr-device cuda
```

- G 抓取后放置；N 跳过当前阶段。
- 抬头后 C 开始外呼识别，U/D 选择上下行。
- 人工进入轿厢并站稳后 C 继续，输入楼层数字，停顿后自动提交，无需回车。
- 楼层阶段已确认的缓存持续保留；`memory=[...]` 列出缓存，R 清空。执行前仍核对图像、深度和位姿。
- 按完楼层键或按 N 跳过后，机械臂进入 LCD 观察姿态。按 C 开始识别和计时，等待 C 或识别期间按 N 回零退出。
- Q 正常回零退出；空格/ESC 停止，不追加自动回零。

完整操作、失败回零和缓存边界说明见 [操作指南](docs/journey_guide.txt)。

## 参数集中修改

`config/261105_my_elevator.yaml`：

- `initial_poses.grasp_ready`：夹取预备位，XYZ 米、RPY 弧度、duration 秒。
- `initial_poses.call_view_deg`、`floor_view_deg`、`display_view_deg`：六轴绝对角度，单位度。
- `search_timeout_s` 分别设置外呼、楼层按键和 LCD 单轮搜索时间，当前为 10、15、10 秒。
- `search_retry_delta_deg` 设置每次超时抬头的 [J2, J3] 相对增量。当前从实测角度减 15°、减 25°，最多重试两次，各轴按 URDF 限位缩短动作。
- `grasp_place.xyz_m`：抓取后放置点，base 坐标系 end_link XYZ，米。
- `elevator.target_offset_base_m`：按钮目标经验补偿。
- `elevator.floor_cache_until_update`：当前总流程的楼层持续缓存开关。

`config/261113_my_journey.yaml` 管理停稳检查、到站超时和 LCD 识别阈值。
配置保留当前机器的实测调试值；坐标、工具偏移和补偿需结合实际安装核对。
软件到达估计表面不等于已经独立确认实体按钮触发，离线测试不替代实机可达性和碰撞检查。

## 发布内容

只保留运行源码、当前配置、测试、测试必需的 8 张照片、控制 SDK 与使用说明。
不包含旧试验备份、终端记录、调试截图、IDE 配置、缓存、凭据、模型二进制或本机标定。
第三方来源与授权边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。
