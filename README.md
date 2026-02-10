<img width="1466" height="1004" alt="c07e0a62a10b1d0aece9321a7f993112" src="https://github.com/user-attachments/assets/58efa08d-b0d3-43be-af12-697eaa23861a" />
<img width="1466" height="1004" alt="c07e0a62a10b1d0aece9321a7f993112" src="https://github.com/user-attachments/assets/c4941965-70f8-4ecf-9a86-891ebf9c7acb" />

 # TeslaCamViewer-Py

本项目是本地桌面版 TeslaCam 查看器（PySide6 + python-vlc），并集成智能事件分析闭环：

- 扫描并分组 TeslaCam 事件（按秒 / 按分钟）
- 本地视觉分析（YOLO + 可解释规则）
- 仅对可疑事件调用 Qwen-VL（DashScope）
- 分析结果写入 SQLite 缓存
- 界面展示、筛选、回放、证据跳转

## 1. 安装

```bash
pip install -r requirements.txt
```

Windows 还需要安装 VLC（并与 Python 位数一致），否则无法播放视频。

## 2. 运行

```bash
python app.py
```

## 3. DashScope 配置

支持两种方式：

1. 在界面顶部点击“智能设置”保存（推荐）
2. 使用环境变量

```powershell
$env:DASHSCOPE_API_KEY="你的密钥"
$env:DASHSCOPE_ENDPOINT="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

国际站地址可改为：

- `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`

## 4. 智能分析流程

1. 对每个事件先做本地分析（默认 10 秒窗口、2 FPS）
2. YOLO 检测人/车/动物
3. 规则引擎产出：主类型、风险等级、对象计数、证据点
4. 仅对高风险候选调用 Qwen-VL，补充高级语义
5. 写入 SQLite，后续相同哈希直接命中缓存

## 5. SQLite 表

程序启动会自动建表/迁移：

- `clips`
- `events`
- `ai_results`（`UNIQUE(event_id, version)`）
- `roi_configs`

## 6. 界面功能

- “开始分析”：后台分析当前目录全部事件（可取消）
- 左侧事件列表显示：主类型、风险、对象计数
- 筛选：
  - 主类型多选
  - 最低风险等级
  - 仅显示已分析事件
- “跳转证据”：跳到证据对应时间点
- “区域设置”：按摄像头设置相对坐标矩形（x/y/w/h，0-1）

## 7. 主要配置项（config.py）

- `AI_SAMPLING_FPS`（默认 2.0）
- `AI_WINDOW_SECONDS`（默认 10）
- `AI_MAX_IMAGES`（默认 12）
- `AI_ENABLE_YOLO`
- `AI_YOLO_MODEL`
- `AI_PASSBY_MAX_SECONDS`
- `AI_LOITER_SECONDS`
- `AI_CLOSE_AREA_GROWTH`
- `AI_LIGHTING_CHANGE_THRESHOLD`
- `AI_IMPACT_DIFF_THRESHOLD`
- `AI_QWEN_TRIGGER_MIN_RISK`

## 8. 常见问题

1. `Could not find libvlc.dll`
- 安装 VLC，并确保 VLC 与 Python 位数一致。

2. 未设置 DashScope 密钥
- 仍可运行本地规则分析，不会调用 Qwen。

3. 事件很多、分析较慢
- 首次分析较慢；后续命中 SQLite 缓存会明显加快。

