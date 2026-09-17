# ops-math

English | [简体中文](./README.md)

## 🔥Latest News

- [2026/01] Added [QuickStart](./QUICKSTART_en.md) to guide beginners with zero foundation to get started with operator project deployment (supporting Docker environments), operator development, and contribution processes.
- [2025/12] Open source operators support Ascend 950PR/Ascend 950DT, which can be developed and debugged through the CANN Simulator simulation tool; added a <<<>>> kernel heterogeneous call example in the add operator for user-defined usage; added support for operators [concat](conversion/concat/), [lerp](math/lerp/), [drop_out_v3](random/drop_out_v3/), and others in multiple categories.
- [2025/11] Improved multiple operator README descriptions and enhanced operator development example documentation and examples.
- [2025/10] Added the experimental directory and improved the [Contribution Guide](CONTRIBUTING_en.md) to support developers in debugging and contributing custom operators.
- [2025/09] The ops-math project was first launched.

## 🚀Overview

ops-math is a basic operator library that provides numerical computation in the [CANN](https://hiascend.com/software/cann) (Compute Architecture for Neural Networks) operator library, including conversion, math, and random categories, covering tensor shape transformation, basic mathematical operations, random number generation, and other scenarios.
<!--
The position of the sub-library in the architecture diagram is as follows.

<img src="docs/zh/figures/architecture.png" alt="Architecture Diagram"  width="700px" height="320px">
-->
This repository has integrated a code repository intelligent agent. Click the [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/ops-math) badge to enter its dedicated page and start an online intelligent code learning and knowledge Q&A experience!

## 📌Version Compatibility

The source code of this project will be released with the CANN software version. For the correspondence between CANN software versions and this project's tags, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure your source code customization development proceeds smoothly, select the compatible CANN version and Gitcode tag source code. Using the master branch may pose a risk of version mismatch.

## 🛠️Environment Preparation

[Environment deployment](docs/en/install/quick_install.md) is a prerequisite for experiencing this project's capabilities. First, complete the NPU driver and CANN package installation to ensure the environment is normal.

## ⬇️Source Code Download

After the environment is ready, download the branch source code compatible with the CANN version. The general command is as follows, replacing $\{tag\_version\} with the branch tag name. Taking the 9.0.0 branch source code download as an example:

```bash
# General command: git clone -b ${tag_version} https://gitcode.com/cann/ops-math.git
git clone -b 9.0.0 https://gitcode.com/cann/ops-math.git
```

> Note: If the compatible branch source code already exists in the environment, **you can skip this step**. For example, CANNLab provides the source code corresponding to the latest CANN version by default.

## 📖Learning Tutorials

- [Quick Start](docs/QUICKSTART_en.md): Quickly experience the project's core basic capabilities from scratch, covering source code compilation, operator invocation, development, and debugging operations.
- [Advanced Tutorials](docs/README_en.md#advanced-tutorials): For in-depth understanding of project compilation deployment, operator invocation, development, debugging, and tuning capabilities, refer to the documentation center for detailed guidance.

## 💬Related Information

- [Directory Structure](docs/en/install/dir_structure.md)
- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [Affiliated SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

-----
PS: This project's functionality and documentation are continuously being updated and improved. We welcome you to follow the latest version.

- **Issue Feedback**: Submit issues through GitCode [【Issues】](https://gitcode.com/cann/ops-math/issues).
- **Community Interaction**: Participate in discussions through GitCode [【Discussions】](https://gitcode.com/cann/ops-math/discussions).
- **Technical Column**: Access technical articles through GitCode [【Wiki】](https://gitcode.com/cann/ops-math/wiki).
