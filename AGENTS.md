# 代码维护说明

## 项目范围

专注于作者改造后的 YingMusic-SVC + Pupu + PC-NSF-HiFiGAN 翻唱方案。环境安装和模型下载由用户按 README 完成，转换服务负责本地推理。

网页以文件输入、参数、进度和结果操作为主。安装和使用说明集中在根目录 README.md。

用户文档直接说明功能、要求和操作步骤，以“有什么、怎么用”为主。实现限制、被移除的功能和本机历史信息留在维护上下文中；涉及用户操作的必要条件，用明确的准备要求表达。

## 目录

- app/server.py 管理服务和任务，app/runner.py 在独立子进程中执行推理。
- inference/ 放推理源码，inference/pupu/ 保留声码器源码及许可证。
- models/、web/、outputs/、data/、logs/ 是根目录下的同级目录。
- app/runtime_paths.py 和 inference/model_paths.py 管理默认位置及环境变量；路径从源码位置计算，不依赖启动时的工作目录。
- 改目录结构时同步修改代码、启动入口、README、检查脚本和 .gitignore。
- 模型、环境、缓存、用户音频、结果和日志不能提交到 Git。清理目录前区分用户数据与测试数据。

## 验证

当前验证环境是 Windows / Python 3.12。先单独安装 PyTorch，再按 requirements.txt 和 constraints.txt 安装其余依赖。

app/check_environment.py 检查依赖、全部必需模型、Whisper 缓存和实际 CUDA 运算，输出检查结果。

涉及路径或推理的改动，用短音频完成一次实际转换，检查进度和输出；不能仅凭网页能打开判断可用。检查文件读取、历史记录、播放和具体任务的结果目录。验证失败要说明实际原因。

## 上游与行为

README.md 和 THIRD_PARTY.md 保留 Stareven233/YingMusic-SVC、GiantAILab/YingMusic-SVC、Pupu 和 openvpi 的引用，保留源码许可证并说明本地适配。

服务仅监听本机。保留取消功能和错误详情；打开结果目录时打开具体任务的输出文件夹。

结果目录使用本地时间戳命名。最近的转换在页面加载或刷新列表时从 outputs/ 读取；当前任务状态保存在内存。
