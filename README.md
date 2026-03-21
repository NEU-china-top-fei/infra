# AI Infra Engineering Tutorial 2026

从后端 / 系统基础走向 **AI 基础设施工程师**（大规模训练、高性能网络、推理引擎、集群调度）的自学路线与笔记仓库。

**Languages:** [中文](README.md) · [English](README.en.md)

## 从这里开始

1. 阅读 [入门与环境](docs/getting-started.md)（前置技能、机器与工具）。
2. 从 **SYS-601** 开课：[courses/SYS-601-gpu-architecture/README.md](courses/SYS-601-gpu-architecture/README.md)。
3. 笔记写入各课的 `notes/`，可复现的硬核产出放在 `projects/` 与 `interview-prep/`。

## 文档目录

| 文档 | 说明 |
|------|------|
| [docs/talent-profile.md](docs/talent-profile.md) | 人才画像、技能矩阵、面试场景与执行建议 |
| [docs/curriculum.md](docs/curriculum.md) | 全年培养方案、SYS-601–605 精读与阅读表 |
| [docs/interview-matrix.md](docs/interview-matrix.md) | 七轮面试与课程映射 |
| [docs/references.md](docs/references.md) | 参考文献与外链索引 |
| [docs/archive/README-full-original.md](docs/archive/README-full-original.md) | 历史单文件版 README（便于对照） |

## 五门核心课

| 课程 | 主题 |
|------|------|
| [SYS-601](courses/SYS-601-gpu-architecture/README.md) | GPU 架构、CUDA/Triton、算子与内存层次 |
| [SYS-602](courses/SYS-602-distributed-training/README.md) | 分布式训练、混合并行、Megatron/ZeRO 等 |
| [SYS-603](courses/SYS-603-networking/README.md) | 高性能网络、NCCL、RDMA/RoCE |
| [SYS-604](courses/SYS-604-inference/README.md) | LLM 推理、KV Cache、PagedAttention / vLLM |
| [SYS-605](courses/SYS-605-cluster-scheduling/README.md) | 集群调度、容错、Checkpoint |

## 学习时间轴

```
 Month 1-3    │ Month 4-7           │ Month 8-10   │ Month 11-12
──────────────┼─────────────────────┼──────────────┼────────────
   SYS-601    │ SYS-602 + SYS-603   │   SYS-604    │   SYS-605
 GPU/CUDA     │ Distributed + Net   │  Inference   │  Scheduling
   (串行)     │      (并行)         │   (串行)     │   (串行)
```

## 仓库结构

```
ai-infra-engineering-tutorial-2026/
├── docs/                         # 完整大纲、人才画像、面试与参考文献
├── courses/                      # SYS-601–605：readings / notes / code / labs
├── interview-prep/               # 七轮面试：round-1 … round-7
├── projects/                     # 实战项目（Portfolio）
└── resources/                    # 论文、博客存档、工具脚本（本地资源见 .gitignore 说明）
```

## 克隆与进度

```bash
git clone https://github.com/<YOUR_GITHUB>/<YOUR_FORK>.git
cd ai-infra-engineering-tutorial-2026
```

建议使用 GitHub **Issues** / **Projects** 按课程打标签追踪进度；面试准备可复用 [interview-prep/TEMPLATE.md](interview-prep/TEMPLATE.md)。Python 实验依赖见根目录 [requirements-labs.txt](requirements-labs.txt)（按需安装）。

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[LICENSE](LICENSE)（Apache-2.0）· [NOTICE](NOTICE)
