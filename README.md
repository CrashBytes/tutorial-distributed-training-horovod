# Distributed ML Training with Horovod and PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![Horovod 0.28](https://img.shields.io/badge/Horovod-0.28-green.svg)](https://github.com/horovod/horovod)

Production-ready distributed training system using Horovod and PyTorch for multi-GPU environments. This repository accompanies the [CrashBytes tutorial](https://crashbytes.com/tutorial-distributed-ml-training-horovod-pytorch-multi-gpu-2025) on building enterprise-scale distributed training systems.

## 🎯 Features

- **Data-Parallel Training**: Efficient multi-GPU training with Horovod's ring-allreduce
- **Mixed Precision**: FP16 training for 2-3x performance improvements
- **Gradient Compression**: Reduce network bandwidth with minimal accuracy impact
- **Fault Tolerance**: Robust checkpointing and recovery mechanisms
- **MLflow Integration**: Comprehensive experiment tracking
- **Production Ready**: Docker containers, configuration management, verification scripts

## 📊 Performance Benchmarks

| GPUs | Batch Size | Throughput | Scaling Efficiency |
|------|------------|------------|-------------------|
| 1    | 128        | 340 img/s  | 100%             |
| 2    | 256        | 665 img/s  | 98%              |
| 4    | 512        | 1,310 img/s| 96%              |
| 8    | 1024       | 2,580 img/s| 95%              |

*Benchmarked on NVIDIA V100 GPUs with ResNet-50 on ImageNet*

## 🚀 Quick Start

See [QUICKSTART.md](docs/QUICKSTART.md) for a 15-minute setup guide.

### Installation

```bash
# Clone repository  
git clone https://github.com/crashbytes/tutorial-distributed-training-horovod.git
cd tutorial-distributed-training-horovod

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]==0.28.1
pip install -r requirements.txt

# Verify setup
python scripts/verify_setup.py
```

### Basic Training

```bash
# Single-node (8 GPUs)
horovodrun -np 8 -H localhost:8 python -m src.training.train --config configs/training_config.yaml

# Multi-node (24 GPUs across 3 nodes)
horovodrun -np 24 -H node1:8,node2:8,node3:8 python -m src.training.train --config configs/training_config.yaml
```

## 📁 Repository Structure

```
.
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading
│   ├── training/          # Training logic
│   └── utils/             # Utilities
├── configs/               # YAML configurations
├── scripts/               # Helper scripts
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── Dockerfile            # Container definition
```

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started in 15 minutes
- [CrashBytes Tutorial](https://crashbytes.com/tutorial-distributed-ml-training-horovod-pytorch-multi-gpu-2025) - Complete walkthrough
- [Horovod Documentation](https://horovod.readthedocs.io/) - Official docs

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

**Michael Eakins** - [CrashBytes](https://crashbytes.com) - [@crashbytes](https://github.com/crashbytes)

---

⭐ **Star this repo if you find it helpful!**
