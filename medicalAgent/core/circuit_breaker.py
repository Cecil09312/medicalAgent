"""
熔断器模块（Circuit Breaker）

用于在大模型（主模型）连续调用失败时自动"熔断"，让请求降级走小模型，
避免持续向故障的大模型发送无效请求。实现经典的三态状态机，语义如下：

- closed（闭合）：正常放行大模型请求；连续失败次数达到阈值后迁移到 open。
- open（断开）：直接拒绝大模型请求（调用方降级走小模型）；冷却时间结束后
  迁移到 half_open。
- half_open（半开）：同一时刻只放行一个探测请求验证大模型是否恢复；
  探测成功迁移回 closed，探测失败重新迁移回 open。

所有状态迁移均通过 asyncio.Lock 串行化，保证并发请求下的状态一致性。
"""
import asyncio
import os
import time
from typing import Dict

from loguru import logger

# 熔断器三态常量
STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

# 默认配置值
_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_COOLDOWN_SECONDS = 60


def _read_int_env(env_name: str, default_value: int) -> int:
    """
    从环境变量读取整数配置，解析失败时使用默认值并记录警告。

    Args:
        env_name: 环境变量名称
        default_value: 解析失败时使用的默认值

    Returns:
        解析后的整数值（解析失败则为默认值）
    """
    raw_value = os.getenv(env_name, str(default_value))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning(f"环境变量 {env_name} 取值无效：{raw_value!r}，使用默认值 {default_value}")
        return default_value


class CircuitBreaker:
    """
    大模型熔断器（三态状态机：closed / open / half_open）

    LLMClient 为每个 Agent 独立实例化，因此熔断器实例必须通过
    get_circuit_breaker 按主模型名跨实例共享，避免各 Agent 的
    失败计数互相隔离导致熔断失效。
    """

    def __init__(self, model_name: str):
        """
        初始化熔断器。

        Args:
            model_name: 绑定的主模型名称（用于日志与单例缓存键）
        """
        self.model_name = model_name
        # 连续失败阈值：达到该次数后熔断（环境变量 CIRCUIT_FAILURE_THRESHOLD，默认 3）
        self.failure_threshold = _read_int_env("CIRCUIT_FAILURE_THRESHOLD", _DEFAULT_FAILURE_THRESHOLD)
        # 冷却秒数：熔断后经过该时长才允许半开探测（环境变量 CIRCUIT_COOLDOWN_SECONDS，默认 60）
        self.cooldown_seconds = _read_int_env("CIRCUIT_COOLDOWN_SECONDS", _DEFAULT_COOLDOWN_SECONDS)

        # 当前状态，初始为闭合
        self._state = STATE_CLOSED
        # 连续失败计数
        self.consecutive_failures = 0
        # 进入 open 状态的时间戳（monotonic 时钟），初始 0.0
        self.opened_at = 0.0

        # 串行化状态迁移的异步锁
        self._lock = asyncio.Lock()
        # 半开探测请求占用标记：同一时刻只放行一个探测请求
        self._probe_in_use = False

    @property
    def state(self) -> str:
        """返回当前状态字符串（closed / open / half_open）"""
        return self._state

    async def allow_request(self) -> bool:
        """
        判断当前是否允许使用大模型（异步加锁保证并发安全）。

        Returns:
            True 表示允许请求大模型；False 表示应降级走小模型
        """
        async with self._lock:
            # 闭合状态：正常放行
            if self._state == STATE_CLOSED:
                return True

            # 断开状态：冷却结束才迁移到半开并放行探测请求
            if self._state == STATE_OPEN:
                now = time.monotonic()
                if now - self.opened_at >= self.cooldown_seconds:
                    self._state = STATE_HALF_OPEN
                    self._probe_in_use = True
                    logger.info(f"大模型 {self.model_name} 熔断冷却结束，进入半开探测状态，放行本次请求作为探测")
                    return True
                # 仍在冷却期内，拒绝大模型请求（调用方降级走小模型）
                return False

            # 半开状态：同一时刻只放行一个探测请求，并发其余请求走小模型
            if not self._probe_in_use:
                self._probe_in_use = True
                return True
            return False

    async def record_success(self) -> None:
        """记录大模型调用成功，必要时将熔断器恢复到闭合状态。"""
        async with self._lock:
            previous_state = self._state
            self.consecutive_failures = 0
            if previous_state == STATE_HALF_OPEN:
                # 半开探测成功，恢复闭合
                self._state = STATE_CLOSED
                self._probe_in_use = False
                logger.info("大模型探测成功，熔断器恢复闭合")
            # closed 状态仅需重置连续失败计数（已在上方完成）

    async def record_failure(self) -> None:
        """记录大模型调用失败，必要时触发或维持熔断。"""
        async with self._lock:
            # 无论何种状态，失败即释放半开探测占用标记
            self._probe_in_use = False

            if self._state == STATE_HALF_OPEN:
                # 半开探测失败，重新熔断
                self._state = STATE_OPEN
                self.opened_at = time.monotonic()
                logger.warning("大模型半开探测失败，重新熔断")
            elif self._state == STATE_CLOSED:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_threshold:
                    self._state = STATE_OPEN
                    self.opened_at = time.monotonic()
                    logger.warning(
                        f"大模型 {self.model_name} 连续失败 {self.consecutive_failures} 次，"
                        f"达到阈值 {self.failure_threshold}，熔断器已断开进入降级，"
                        f"冷却 {self.cooldown_seconds} 秒后再尝试半开探测"
                    )
            # open 状态下无需额外处理


# 进程内按主模型名缓存的熔断器单例表。
# LLMClient 为每 Agent 独立实例化，熔断计数必须跨实例共享，
# 因此通过该缓存保证同一主模型名全局只有一个熔断器实例。
_breakers: Dict[str, "CircuitBreaker"] = {}


def get_circuit_breaker(model_name: str) -> "CircuitBreaker":
    """
    获取（或创建）与主模型名绑定的熔断器单例。

    Args:
        model_name: 主模型名称，作为进程内缓存键

    Returns:
        该模型名对应的 CircuitBreaker 实例（进程内共享）
    """
    breaker = _breakers.get(model_name)
    if breaker is None:
        breaker = CircuitBreaker(model_name)
        _breakers[model_name] = breaker
    return breaker
