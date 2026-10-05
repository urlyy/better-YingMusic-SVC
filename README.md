# better-YingMusic

本地 AI 翻唱工具：输入原唱人声和参考声音，在网页里把原唱换成参考音色。

推理基于 [Stareven233/YingMusic-SVC](https://github.com/Stareven233/YingMusic-SVC) 的改造版，采用 **YingMusic-SVC → Pupu-Vocoder → PC-NSF-HiFiGAN** 方案。YingMusic 原项目是 [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC)。本项目提供网页、文件管理和任务进度，更多来源见 [THIRD_PARTY.md](THIRD_PARTY.md)，上游许可证保留在源码目录里。

仓库包含网页、服务器和推理代码。clone 后自行安装 Python 依赖、下载模型，再启动服务。以下是 **Windows PowerShell** 的操作步骤，所有命令都在项目根目录执行。

## 功能

- 通过文件选择、拖放或本机路径读取原唱人声和参考音色。
- 试听输入音频，调整音高、推理步数、计算精度和引导强度。
- 查看转换阶段与推理进度，随时取消任务。
- 试听和下载转换结果，打开对应任务的结果目录。
- 浏览最近的转换，点击记录打开对应结果目录。

## 一键启动

首次使用，按下文完成依赖安装和模型下载，并运行环境检查。

启动脚本使用项目里的 `.venv` 环境。准备好后，**双击项目根目录的 [启动.cmd](启动.cmd)**，即可启动服务并自动打开网页。以后每次使用都可以直接双击这个文件。

启动窗口保持打开，关闭窗口即可停止服务。页面地址是 [http://localhost:23335](http://localhost:23335)。

## 目录

```text
better-YingMusic-SVC/
├── app/                      # 网页服务、推理调用、路径配置和环境检查
├── inference/                # 推理代码，pupu/ 内是声码器源码
├── models/                   # 下载的模型
├── web/                      # 网页界面
├── outputs/                  # 转换结果
├── data/                     # 上传文件
├── logs/                     # 转换日志
├── requirements.txt          # Python 依赖
├── constraints.txt           # 已验证的 PyTorch 版本
├── 启动.cmd      # Windows 启动入口
├── README.md                 # 安装和使用说明
├── AGENTS.md                 # 代码维护说明
└── THIRD_PARTY.md            # 上游来源说明
```

## 1. 安装依赖

运行环境：**Windows、Python 3.12、支持 CUDA 的 NVIDIA 显卡**。先安装 Python 和显卡驱动，再执行下面的命令。

```powershell
git clone https://github.com/urlyy/better-YingMusic-SVC.git
cd better-YingMusic-SVC
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install -r requirements.txt -c constraints.txt
```

这组依赖已在 RTX 5060 Ti 上完成推理验证。其他显卡如果需要不同的 CUDA 构建，按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择，并相应调整 `constraints.txt` 的版本。先安装 PyTorch，再安装其余依赖。

安装和运行统一使用项目虚拟环境中的 `.venv/Scripts/python.exe`。

## 2. 下载模型

打开下列链接下载对应文件。**右边的路径都从项目根目录开始算**，按表格创建目录并放入文件。

| 下载什么 | 下载链接 | 放在哪里 |
| --- | --- | --- |
| YingMusic 主模型 | [YingMusic-SVC-full.pt](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt) | `models/YingMusic-SVC-full.pt` |
| CAMPPlus 音色模型 | [campplus_cn_common.bin](https://huggingface.co/funasr/campplus/blob/main/campplus_cn_common.bin) | `models/campplus_cn_common.bin` |
| RMVPE 音高模型 | [model.pt](https://huggingface.co/Pur1zumu/RIFT-SVC-modules/blob/main/rmvpe/model.pt) | `models/rmvpe/model.pt` |
| Pupu 小版 generator | [指定 checkpoint 文件夹](https://huggingface.co/spellbrush/AliasingFreeNeuralAudioSynthesis/tree/main/pupuvocoder/checkpoint/epoch-0051_step-2553605_loss-62.135194) 中的 `model.safetensors` | `models/pupu/model.safetensors` |
| PC-NSF-HiFiGAN | [2025.02 发布页](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02) 的模型 ZIP | 解压，取出 `config.json` 和 `model.ckpt`，放进 `models/pc_nsf_hifigan/` |

Pupu 使用上表指定 checkpoint 中的 `model.safetensors`，对应源码位于 `inference/pupu/`。PC 发布包中的许可证和声明请一并保留。

还需要 Whisper。装完依赖后执行下面这条命令，它会下载需要的三个文件：

```powershell
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small', cache_dir='models/huggingface/hub', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
```

来源是 [openai/whisper-small](https://huggingface.co/openai/whisper-small)。保留下载工具生成的缓存目录结构。

准备好后应当是：

```text
models/
├── YingMusic-SVC-full.pt
├── campplus_cn_common.bin
├── rmvpe/model.pt
├── pupu/model.safetensors
├── pc_nsf_hifigan/
│   ├── config.json
│   └── model.ckpt
└── huggingface/hub/          # Whisper 缓存
```

准备完上述模型后，转换时会从本地加载。

## 3. 检查并启动

先检查环境。缺少包或模型时，会打印具体名称或路径：

```powershell
.venv/Scripts/python.exe app/check_environment.py
```

检查通过后，双击根目录的 **`启动.cmd`** 一键启动。

也可以在终端启动：

```powershell
.venv/Scripts/python.exe app/server.py
```

浏览器会自动打开 [http://localhost:23335](http://localhost:23335)。

服务运行时保留启动窗口，关闭窗口即可停止服务。手动打开网页时，可使用 `.venv/Scripts/python.exe app/server.py --no-browser` 启动。

## 4. 怎么翻唱

1. **原唱人声**：选已经分离出的人声音频。可以选文件、拖入文件或粘贴本机路径。先试听，确认选对了。
2. **参考音色**：选想换成的声音。优先用干净的单人清唱，少伴奏、少混响；10–25 秒即可，超过 25 秒只取开头。说话音频也能尝试，翻唱优先使用演唱片段。
3. **音高**：默认 0。男声转女声可以试升调，女声转男声可以试降调，一格是一个半音。按歌曲和目标声线试听调整。
4. **开始转换**：先用默认的 30 步、BF16、引导强度 0.7。右侧显示当前阶段和进度，需要时可以取消。
5. **听结果**：完成后直接试听、下载或打开这次任务的结果文件夹。

输出是 44.1 kHz FLAC，保存在 `outputs/2026-10-06_14-30-05/`（按转换开始时间命名）。将转换后的人声与歌曲伴奏混合，即可得到完整翻唱。

## 文件和记录

上传或拖入的文件复制到 `data/`；粘贴路径读取直接使用原文件。浏览器记住最近选择。打开页面或点击“刷新列表”时，“最近的转换”读取 `outputs/` 中的结果；删除对应结果目录后，刷新列表即可更新。刷新页面后可以继续查看正在运行的任务。

转换进度包括模型加载、音频分析、音色转换、声音合成和保存。音色转换阶段按实际推理步数更新，其他阶段显示当前状态。

`outputs/` 放结果，`logs/` 放日志，`data/` 放上传文件。最近的结果直接从输出目录读取。清理前先停止服务，再按需删除文件。

## 常见问题

| 问题 | 处理方法 |
| --- | --- |
| 缺少 Python 包 | 用上面的 `.venv/Scripts/python.exe` 安装依赖，再运行环境检查。 |
| 找不到模型 | 对照模型表检查文件名、目录层级、下载是否完整。确认 PC 的两个文件直接位于 `models/pc_nsf_hifigan/`。 |
| 找不到 Whisper 缓存 | 执行 Whisper 下载命令；如果设置过 `HF_HUB_CACHE`，检查是否指向了其他目录。 |
| CUDA 不可用 | 检查驱动，确认安装的是 CUDA 版 PyTorch。环境检查会实际运行一次显卡计算。 |
| 显存不足 | 试 BF16 或 FP16，先转换短一些的片段，关闭其他占用显存的程序。 |
| 端口被占用 | 检查是否已经启动过服务，打开已有页面，或关闭旧服务后再启动。 |
| 转换失败 | 查看界面的错误详情和 `logs/任务编号.log`，定位具体的失败原因。 |
| 音色效果不好 | 先换干净清唱参考，确认原唱没有明显伴奏残留，再调整音高。 |

## 自定义位置

默认使用项目中的环境和目录。复用其他位置的环境或模型时，在启动前设置环境变量：

| 变量 | 用途 | 默认 |
| --- | --- | --- |
| `PUPU_MODEL_DIR` | 模型目录，内部结构按上表保留 | 项目里的 `models/` |
| `PUPU_ENGINE_DIR` | 推理源码目录 | 项目里的 `inference/` |
| `PUPU_PYTHON` | 执行推理的 Python | 启动服务的 Python |
| `PUPU_HF_CACHE` | Whisper 缓存 | `models/huggingface/hub/` |

也支持 `HF_HUB_CACHE`，同时设置时 `PUPU_HF_CACHE` 优先。例如：

```powershell
$env:PUPU_MODEL_DIR = 'D:/AI/models/better-YingMusic'
.venv/Scripts/python.exe app/check_environment.py
.venv/Scripts/python.exe app/server.py
```

服务只监听本机 `127.0.0.1:23335`，网页资源来自本地。模型授权以各下载来源的说明为准。
