#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/serve/inference.py
======================
模型加载与流式生成（推理服务核心）。

设计要点:
    1. 懒加载：进程启动不加载模型，首次对话时才加载；模型缺失/加载失败时
       服务仍可启动，/health 返回可用状态，/chat/stream 返回可解释错误。
    2. 两种模型：
       - merged（微调后）：优先用 MERGED_MODEL_PATH 合并模型；
         若不存在但基座 + LoRA 适配器齐全，则动态加载基座并挂接 PeftModel。
       - base（微调前）：直接加载 BASE_MODEL_PATH 基座。
    3. ChatML：与训练一致，使用 tokenizer.apply_chat_template，
       并注入与 build_dataset.py 相同的 SYSTEM_PROMPT。
    4. 流式：TextIteratorStreamer + 后台线程，支持 stop_event 中途停止
       （对应前端「停止生成」按钮）。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv

# 加载项目根目录 .env（已 export 的环境变量优先，不覆盖）
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# 与 src/build_dataset.py 中训练数据的 system prompt 保持一致
SYSTEM_PROMPT = (
    "你是一名专业的中文医疗助手，请基于医学知识为患者提供准确、严谨的回答，"
    "并在必要时建议就医。"
)

# ---- 环境变量 ----
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "").strip()
MERGED_MODEL_PATH = os.getenv("MERGED_MODEL_PATH", "./outputs/merged_model").strip()
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./outputs/lora_adapter").strip()
DEVICE_CONF = os.getenv("DEVICE", "auto").strip().lower()

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))


# ---- 设备与路径探测 ----

def resolve_device() -> str:
    """解析推理设备：DEVICE=auto 时按 CUDA 可用性自动选择。"""
    if DEVICE_CONF in ("cuda", "cpu"):
        return DEVICE_CONF
    return "cuda" if torch.cuda.is_available() else "cpu"


def _has_config(path: str) -> bool:
    """判断目录是否为可用的 HF 模型目录（含 config.json）。"""
    return bool(path) and (Path(path) / "config.json").is_file()


def _has_adapter(path: str) -> bool:
    """判断目录是否为可用的 LoRA 适配器目录（含 adapter_config.json）。"""
    return bool(path) and (Path(path) / "adapter_config.json").is_file()


def model_availability() -> dict:
    """返回各模型来源的可用状态，供 /health 展示。"""
    merged_ok = _has_config(MERGED_MODEL_PATH)
    base_ok = _has_config(BASE_MODEL_PATH)
    lora_ok = _has_adapter(LORA_ADAPTER_PATH)
    return {
        "device": resolve_device(),
        "base": {"path": BASE_MODEL_PATH or "(未配置)", "available": base_ok},
        "merged": {"path": MERGED_MODEL_PATH, "available": merged_ok},
        "lora": {"path": LORA_ADAPTER_PATH, "available": lora_ok},
        # 微调后可用 = 合并模型存在，或基座+适配器齐全
        "finetuned_ready": merged_ok or (base_ok and lora_ok),
    }


# ---- 停止条件 ----

class _StopOnEvent(StoppingCriteria):
    """外部事件触发时立即停止生成（前端「停止」按钮 / 客户端断开）。"""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


# ---- 模型管理器（懒加载 + 单例缓存） ----

class ModelManager:
    """按 model_key 缓存已加载的 (model, tokenizer, label)，线程安全。"""

    _cache: dict[str, tuple] = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, model_key: str):
        """获取模型；已缓存则直接返回，否则加锁加载（只加载一次）。"""
        if model_key in cls._cache:
            return cls._cache[model_key]
        with cls._lock:
            if model_key not in cls._cache:
                # 单卡显存有限：切换模型前先释放另一个已缓存的模型，
                # 避免微调前/后两份 7B 模型同时驻留导致 CUDA OOM
                for old_key in list(cls._cache.keys()):
                    if old_key != model_key:
                        old_model = cls._cache.pop(old_key)[0]
                        del old_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"[MODEL] 已释放模型缓存: {old_key}")
                cls._cache[model_key] = cls._build(model_key)
            return cls._cache[model_key]

    @staticmethod
    def _build(model_key: str):
        """实际加载模型。model_key: 'merged'（微调后）或 'base'（微调前）。"""
        device = resolve_device()
        # GPU 上优先 bf16（不支持则 fp16）；CPU 用 fp32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if model_key == "merged":
            if _has_config(MERGED_MODEL_PATH):
                # 优先：合并后的单一 HF 模型
                path = MERGED_MODEL_PATH
                label = f"微调后(合并模型) {path}"
                print(f"[MODEL] 加载合并模型: {path} (dtype={dtype}, device={device})")
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=dtype, trust_remote_code=True
                )
            elif _has_config(BASE_MODEL_PATH) and _has_adapter(LORA_ADAPTER_PATH):
                # 回退：基座 + LoRA 适配器动态挂接
                print(f"[MODEL] 未找到合并模型，改为基座 + LoRA 适配器: "
                      f"{BASE_MODEL_PATH} + {LORA_ADAPTER_PATH}")
                tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL_PATH, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_PATH, torch_dtype=dtype, trust_remote_code=True
                )
                from peft import PeftModel  # 懒导入：仅在此分支需要
                model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
                label = f"微调后(基座+LoRA) {LORA_ADAPTER_PATH}"
            else:
                raise RuntimeError(
                    "微调后模型不可用：既找不到合并模型 "
                    f"{MERGED_MODEL_PATH}/config.json，也不具备基座 + LoRA 适配器。"
                    "请先执行 bash scripts/03_merge.sh，或检查 .env 中路径配置。"
                )
        elif model_key == "base":
            if not _has_config(BASE_MODEL_PATH):
                raise RuntimeError(
                    f"微调前基座模型不可用：BASE_MODEL_PATH={BASE_MODEL_PATH or '(未配置)'} "
                    "下找不到 config.json，请检查 .env 配置。"
                )
            label = f"微调前(基座) {BASE_MODEL_PATH}"
            print(f"[MODEL] 加载基座模型: {BASE_MODEL_PATH} (dtype={dtype}, device={device})")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH, torch_dtype=dtype, trust_remote_code=True
            )
        else:
            raise ValueError(f"未知 model_key: {model_key}（仅支持 'merged' / 'base'）")

        model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[MODEL] 加载完成: {label}")
        return model, tokenizer, label


# ---- 流式对话 ----

def stream_chat(
    messages: list[dict],
    model_key: str = "merged",
    stop_event: Optional[threading.Event] = None,
) -> Iterator[tuple]:
    """
    流式生成回答（同步生成器，供 app.py 在线程池中迭代）。

    参数:
        messages:   [{"role": "user"/"assistant"/"system", "content": "..."}]
        model_key:  "merged"（微调后）或 "base"（微调前）
        stop_event: 置位后立即停止生成

    yield 事件元组:
        ("token", text)                 逐段生成文本
        ("done", tokens:int, elapsed:float)  正常/停止结束
        ("error", message:str)          加载或生成异常
    """
    try:
        model, tokenizer, label = ModelManager.load(model_key)
    except Exception as e:
        yield ("error", f"模型加载失败：{e}")
        return

    # 补全 system prompt（与训练数据格式一致）
    chat_messages = [m for m in messages if m.get("content") is not None]
    if not chat_messages or chat_messages[0].get("role") != "system":
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_messages

    try:
        prompt_text = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stopping = None
        if stop_event is not None:
            stopping = StoppingCriteriaList([_StopOnEvent(stop_event)])

        gen_result: dict = {}

        def _run_generate():
            try:
                start = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        repetition_penalty=REPETITION_PENALTY,
                        pad_token_id=tokenizer.pad_token_id,
                        stopping_criteria=stopping,
                    )
                gen_result["tokens"] = int(output_ids[0].shape[0] - input_len)
                gen_result["elapsed"] = time.time() - start
            except Exception as e:  # 生成期异常传回主迭代循环
                gen_result["error"] = f"{type(e).__name__}: {e}"

        worker = threading.Thread(target=_run_generate, daemon=True)
        worker.start()

        # streamer 迭代会阻塞到生成结束（或被 stop_event 终止）
        for chunk in streamer:
            if chunk:
                yield ("token", chunk)

        worker.join()
        if "error" in gen_result:
            yield ("error", gen_result["error"])
        else:
            yield ("done", gen_result.get("tokens", 0), gen_result.get("elapsed", 0.0))
    except Exception as e:
        yield ("error", f"{type(e).__name__}: {e}")
