# 第三方源码

- `inference/`：来自 [Stareven233/YingMusic-SVC](https://github.com/Stareven233/YingMusic-SVC)，fork 自 [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC)。许可证见该目录的 `LICENSE.txt`。
- `inference/pupu/`：来自 [sizigi/AliasingFreeNeuralAudioSynthesis](https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis)，许可证见该目录的 `LICENSE`。
- PC-NSF-HiFiGAN 实现随作者的 YingMusic 改造版提供，来源为 [openvpi/vocoders](https://github.com/openvpi/vocoders)；使用权重时保留发布包内的声明文件。

本仓库使用已验证的上游源码快照。主要适配包括本地 CAMPPlus 权重读取、SoundFile FLAC 导出、Studio 网页界面和结构化进度事件。模型下载与环境安装步骤见 [README.md](README.md)。