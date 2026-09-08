# 高效微调中文医疗问答 Chatbot 🏥

基于 **Qwen2.5-7B-Instruct** + **QLoRA 4-bit**（Unsloth + TRL SFTTrainer）的中文医疗问答 Chatbot。

> 训练：在 6 个科室（内科 / 外科 / 儿科 / 妇产科 / 男科 / 肿瘤科）各 2 万条问答（共 12 万）上做 1 epoch 微调。
> 部署：FastAPI 推理服务 + ChatGPT 风格 Web 界面，支持流式输出、计时、多轮会话、医疗免责声明。

详细方案见 [`PLAN.md`](./PLAN.md)。

---

## 目录结构

```
chineseMedicalLora/
├── 素材/                              # 既有材料（保留作参考）
│   ├── Qwen2_5_(7B)_医疗微调.ipynb
│   └── Qwen2_5_(7B)_医疗微调.py
├── datasets/                          # 中文医疗问答数据（6 科室）
├── src/                               # 核心代码
│   ├── build_dataset.py               # CSV → JSONL 数据预处理
│   ├── train.py                       # Unsloth + LoRA 训练
│   ├── merge_lora.py                  # LoRA → 合并 HF 模型
│   └── serve/
│       ├── app.py                     # FastAPI 入口
│       ├── inference.py               # 模型加载 + 生成
│       └── static/                    # 前端
│           ├── index.html
│           ├── style.css
│           ├── app.js
│           └── image.png              # 界面预览截图
├── scripts/                           # 一键脚本
│   ├── 01_build_data.sh
│   ├── 02_train.sh
│   ├── 03_merge.sh
│   └── 04_serve.sh
├── data/                              # 处理后数据
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── _dryrun.jsonl
│       └── stats.json
├── outputs/                           # 训练产物
│   ├── lora_adapter/                  # LoRA 权重 ~100MB
│   ├── merged_model/                  # 合并后 HF 模型 ~15GB
│   └── logs/
├── PLAN.md                            # 完整方案文档
├── requirements.txt
└── .env.example
```

---

## 快速开始（4 步）

> 推荐环境：AutoDL 容器 / 任意 Linux + GPU ≥ 24GB（RTX 4090 / A5000 / A100）。CPU 可跑推理但慢。

### 0. 创建 Conda 环境 + 安装依赖

**本机已验证环境：PyTorch 2.5.1 / Python 3.11 / CUDA 12.4 / RTX 4090 24GB (Ubuntu 22.04)**

```bash
# 1) 克隆/拷贝项目到本地（AutoDL 上通常 /root/autodl-tmp/work/）
cd /root/autodl-tmp/work
# 假设项目已放在 chineseMedicalLora/ 下
cd chineseMedicalLora

# 2) 创建并激活 conda 环境（Python 3.11 对 Unsloth / bitsandbytes 兼容性最稳）
conda create -n medical-sft python=3.11 -y
conda activate medical-sft

# 3) 安装 PyTorch（必须 cu124：驱动 CUDA≤12.6 时勿装官方默认的 cu130）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 4) 安装项目依赖（会钉死 torch，避免被 unsloth 升级）
pip install -r requirements.txt
# 若环境里已有 torchao，请卸掉（与 torch 2.5.1 不兼容）:
# pip uninstall -y torchao

# 5) 验证 GPU 可用
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
# 预期: PyTorch: 2.5.1+cu124 | CUDA: 12.4 | device: NVIDIA GeForce RTX 4090

# 6) 配置环境变量（脚本与 Python 入口会自动加载 .env）
cp .env.example .env
# 编辑 .env，至少确认 BASE_MODEL_PATH 指向含 config.json 的基座目录
```

> **AutoDL 用户注意**：驱动常见为 CUDA 12.6（如 560.x）。务必先装 `torch==2.5.1+cu124`，再装 requirements；不要让 pip 把 torch 升到 `cu128/cu130`。若 `pip install unsloth` 仍报错，参考 §常见问题 Q7。

### 1. 构建数据集

```bash
bash scripts/01_build_data.sh
# 输出: data/processed/{train,val}.jsonl + stats.json
# 默认每科 20000 条，共 ~117600 train + ~2400 val
```

### 2. 训练（**先 dry-run 再正式**）

```bash
# (a) Dry-run：用 200 条数据跑 10 步验证流水线（~3 分钟）
bash scripts/02_train.sh --dry-run

# (b) 正式训练：方案 B - 2万/科 × 1 epoch（RTX 4090 约 2.1 小时）
bash scripts/02_train.sh

# 断点续训
bash scripts/02_train.sh --resume-from outputs/lora_adapter/checkpoint-500
```

### 3. 合并 LoRA → HF 模型

```bash
bash scripts/03_merge.sh
# 输出: outputs/merged_model/ （fp16 ~15GB）
```

### 4. 启动推理服务

```bash
bash scripts/04_serve.sh
# 访问 http://localhost:8000
# AutoDL 需用平台提供的"代理"功能或自定义端口映射
```

---

## 端到端验证清单

- [ ] `curl http://localhost:8000/health` → `{"status":"ok","model_loaded":true}`
- [ ] 浏览器打开首页，看到顶置免责声明 banner
- [ ] 点击"我最近总是感觉头晕"示例问题
- [ ] 流式逐字显示回答，底部显示 `⏱ X.XXs · N tokens`
- [ ] 再问"我之前问过头晕，现在还需要注意什么？"（多轮）
- [ ] 流式中途点击"停止"按钮
- [ ] 点击"清空对话"按钮

---

## 关键参数（环境变量）

> 写在项目根目录 `.env` 即可；`scripts/*.sh` 与训练/合并/推理入口会自动加载。

| 变量 | 默认值 / 示例 | 说明 |
| --- | --- | --- |
| `BASE_MODEL_PATH` | `/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master` | 基座模型目录（须含 `config.json`） |
| `MERGED_MODEL_PATH` | `./outputs/merged_model` | 合并后模型路径 |
| `RAW_DATA_DIR` | `./datasets` | 原始数据 |
| `PROCESSED_DATA_DIR` | `./data/processed` | 处理后 JSONL |
| `N_PER_DEPT` | `20000` | 每科采样条数 |
| `MAX_CHAR_LEN` | `600` | ask+answer 字符上限 |
| `MAX_NEW_TOKENS` | `512` | 生成最大 token 数 |
| `TEMPERATURE` | `0.7` | 采样温度 |
| `TOP_P` | `0.9` | nucleus 采样 |
| `REPETITION_PENALTY` | `1.1` | 重复惩罚 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |

---

## 训练超参一览

```python
MAX_SEQ_LENGTH        = 2048
LORA_R                = 16
LORA_ALPHA            = 16
LORA_DROPOUT          = 0
LORA_TARGET_MODULES   = [q,k,v,o,gate,up,down]_proj
LEARNING_RATE         = 2e-4
PER_DEVICE_BATCH      = 2
GRAD_ACCUM            = 4        # effective batch = 8
EPOCHS                = 1
OPTIM                 = adamw_8bit
PACKING               = True     # ~3× 加速
SEED                  = 3407
```

---

## 常见问题

**Q1: 显存不够怎么办？**
- 方案 A：把 `LORA_R` 降到 8（参数更少）。
- 方案 B：把 `MAX_SEQ_LENGTH` 降到 1024（数据本来就短，平均 110 token）。
- 方案 C：把 `PER_DEVICE_BATCH` 降到 1，`GRAD_ACCUM` 保持 4。
- 方案 D：用 `merged_4bit` 保存 + `load_in_4bit=True` 部署。

**Q2: AutoDL 抢不到 4090 怎么办？**
- 退到 A5000（24G），训练时间 +20%。
- 退到 T4（16G）会很慢，不推荐 12 万数据。

**Q3: 训练中断了能续吗？**
- 可以。`bash scripts/02_train.sh --resume-from outputs/lora_adapter/checkpoint-500`
- 每 500 步自动 save，保留最近 2 个 checkpoint。

**Q4: 怎么调整采样量？**
```bash
N_PER_DEPT=10000 bash scripts/01_build_data.sh   # 每科 1 万，总 6 万
N_PER_DEPT=30000 bash scripts/01_build_data.sh   # 每科 3 万，总 18 万
```

**Q5: 模型效果不理想怎么办？**
- 增大数据：把 N_PER_DEPT 调到 30000+。
- 多 epoch：编辑 `src/train.py` 把 `NUM_TRAIN_EPOCHS=1` 改为 2 或 3。
- 改超参：尝试 `LEARNING_RATE=1e-4`（更稳定）。
- 注意：本方案为"快速验证版"，非生产级。

**Q6: Web 界面打不开？**
- 检查端口：`netstat -tlnp | grep 8000`
- 检查防火墙：AutoDL 需在平台控制台开端口。
- 检查浏览器 Console：F12 看是否有 404 / 500。

**Q7: `pip install unsloth` / 训练启动报错怎么办？**
AutoDL 上常见几种情况：
```bash
# 报错 1: "No module named 'torch'"  →  conda 装好但 pip 走的不是 conda 环境
conda activate medical-sft
which pip        # 确认在 .../envs/medical-sft/bin/pip
which python

# 报错 2: "CUDA driver too old" / Unsloth cannot find any torch accelerator
#   → 装成了 torch+cu130，而驱动只到 CUDA 12.6
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28.post3

# 报错 3: AttributeError: module 'torch' has no attribute 'int1'
#   → torchao 过新，与 torch 2.5.1 不兼容（本项目不需要 torchao）
pip uninstall -y torchao

# 报错 4: Can't load configuration / HFValidationError on local path
#   → BASE_MODEL_PATH 未指向含 config.json 的目录；检查 .env 后重跑

# 报错 5: bitsandbytes 找不到 libcudart
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

如果以上都解决不了，**换用 AutoDL 官方/社区训练镜像**（推荐 `PyTorch 2.5.1 + Python 3.11 + CUDA 12.4`）。

---

## 风险与注意

1. **数据来源**：本数据集来自第三方医疗问答网站，部署公网前需评估合规性。
2. **效果边界**：仅 12 万 × 1 epoch，长尾医学问题可能答得泛。生产场景需更多数据 + 多 epoch。
3. **TRL 版本**：训练使用 `SFTConfig`（`trl` 0.12–0.24）；`assistant_only_loss` / `packing` / `dataset_num_proc` 写在 `SFTConfig` 里，不要放进旧的 `TrainingArguments`。
4. **合并后磁盘**：fp16 约 15GB，确保 AutoDL 容器数据盘 ≥ 30GB。
5. **依赖钉死**：勿升级到 `transformers` 5.x 或 `torch` cu130，除非同步升级驱动与整套栈。

---

## 致谢

- Unsloth 团队（训练加速）
- TRL / HuggingFace PEFT（SFT 框架）
- Qwen 团队（基座模型）
