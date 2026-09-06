/* ============================================================
 * chineseMedicalLora 前端交互逻辑
 * 功能：流式问答、多轮会话、停止生成、清空对话、
 *       微调前/微调后模型切换、健康检查、用时与 token 统计
 *
 * 后端接口契约（FastAPI）：
 *   GET  /health        -> { "status": "ok", "model_loaded": true }
 *   POST /chat/stream   <- { "messages": [{role, content}...],
 *                            "model": "merged" | "base" }
 *                        -> SSE 流，每行一条事件：
 *                           data: {"token": "片段"}
 *                           data: {"done": true, "tokens": 128, "elapsed": 3.45}
 *                           data: {"error": "错误信息"}
 *                           data: [DONE]
 * ============================================================ */

(function () {
    'use strict';

    // ---------------- DOM 元素 ----------------
    var chatArea = document.getElementById('chatArea');
    var welcome = document.getElementById('welcome');
    var messageInput = document.getElementById('messageInput');
    var sendBtn = document.getElementById('sendBtn');
    var stopBtn = document.getElementById('stopBtn');
    var clearBtn = document.getElementById('clearBtn');
    var statusDot = document.getElementById('statusDot');
    var statusText = document.getElementById('statusText');
    var modelBtns = document.querySelectorAll('.model-btn');
    var exampleChips = document.querySelectorAll('.example-chip');

    // ---------------- 全局状态 ----------------
    var history = [];          // 多轮会话历史 [{role: 'user'|'assistant', content: '...'}]
    var streaming = false;     // 是否正在流式生成
    var abortController = null; // 用于「停止生成」
    var currentModel = 'merged'; // merged=微调后  base=微调前

    // ---------------- 健康检查 ----------------
    function checkHealth() {
        fetch('/health')
            .then(function (resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
            .then(function (data) {
                var ok = data && data.status === 'ok';
                statusDot.className = 'status-dot ' + (ok ? 'ok' : 'error');
                statusText.textContent = ok
                    ? (data.model_loaded ? '模型已就绪' : '模型加载中')
                    : '服务异常';
            })
            .catch(function () {
                statusDot.className = 'status-dot error';
                statusText.textContent = '服务未连接';
            });
    }

    // ---------------- 工具函数 ----------------
    // 滚动到聊天区域底部
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // 输入框高度自适应
    function autoGrow() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
    }

    // 切换「生成中 / 空闲」的按钮与控件状态
    function setStreaming(on) {
        streaming = on;
        sendBtn.classList.toggle('hidden', on);
        stopBtn.classList.toggle('hidden', !on);
        sendBtn.disabled = on;
        clearBtn.disabled = on;
        modelBtns.forEach(function (btn) { btn.disabled = on; });
    }

    // 创建一条消息行，返回 {row, bubble, meta}
    function createMessageRow(role) {
        var row = document.createElement('div');
        row.className = 'message-row ' + role;

        var avatar = document.createElement('div');
        avatar.className = 'avatar ' + role;
        avatar.textContent = role === 'user' ? '我' : '医';

        var body = document.createElement('div');
        body.className = 'message-body';

        var bubble = document.createElement('div');
        bubble.className = 'bubble';

        var meta = document.createElement('div');
        meta.className = 'message-meta hidden';

        body.appendChild(bubble);
        body.appendChild(meta);
        row.appendChild(avatar);
        row.appendChild(body);
        chatArea.appendChild(row);
        return { row: row, bubble: bubble, meta: meta };
    }

    // ---------------- 发送与流式接收 ----------------
    function sendMessage(text) {
        var content = (text !== undefined ? text : messageInput.value).trim();
        if (!content || streaming) return;

        // 隐藏欢迎页，展示用户消息
        welcome.classList.add('hidden');
        messageInput.value = '';
        autoGrow();

        createMessageRow('user').bubble.textContent = content;
        history.push({ role: 'user', content: content });
        scrollToBottom();

        // 助手空气泡 + 流式光标
        var assistant = createMessageRow('assistant');
        var bubble = assistant.bubble;
        var meta = assistant.meta;
        var cursor = document.createElement('span');
        cursor.className = 'cursor';
        bubble.appendChild(cursor);

        setStreaming(true);
        var startTime = Date.now();
        abortController = new AbortController();

        fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: history, model: currentModel }),
            signal: abortController.signal
        })
            .then(function (resp) {
                if (!resp.ok) throw new Error('服务返回 HTTP ' + resp.status);
                return readStream(resp.body, function (token) {
                    // 每收到一个 token 片段，插入到光标之前
                    var node = document.createTextNode(token);
                    bubble.insertBefore(node, cursor);
                    scrollToBottom();
                });
            })
            .then(function (result) {
                // 正常结束：result 为 done 事件携带的统计信息
                finishStream(assistant, cursor, {
                    tokens: result.tokens,
                    elapsed: result.elapsed,
                    stopped: false
                });
            })
            .catch(function (err) {
                if (err.name === 'AbortError') {
                    // 用户点击「停止」
                    finishStream(assistant, cursor, {
                        tokens: null,
                        elapsed: (Date.now() - startTime) / 1000,
                        stopped: true
                    });
                } else {
                    // 请求或流异常：气泡标红提示
                    cursor.remove();
                    bubble.classList.add('error');
                    bubble.textContent = '请求失败：' + err.message + '。请确认推理服务已启动（/health 正常）后重试。';
                    meta.classList.remove('hidden');
                    meta.textContent = '生成失败';
                    setStreaming(false);
                    abortController = null;
                    scrollToBottom();
                }
            });
    }

    // 读取 SSE 流，对每个 token 回调 onToken；resolve done 事件的数据
    function readStream(stream, onToken) {
        var reader = stream.getReader();
        var decoder = new TextDecoder('utf-8');
        var buffer = '';
        var doneData = { tokens: null, elapsed: null };

        function pump() {
            return reader.read().then(function (chunk) {
                if (chunk.done) return doneData;
                buffer += decoder.decode(chunk.value, { stream: true });

                // SSE 事件以空行分隔，按行解析 data: 前缀
                var lines = buffer.split('\n');
                buffer = lines.pop(); // 最后一段可能不完整，留到下次
                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i].trim();
                    if (line.indexOf('data:') !== 0) continue;
                    var payload = line.slice(5).trim();
                    if (!payload) continue;
                    if (payload === '[DONE]') return doneData;

                    var evt = JSON.parse(payload);
                    if (evt.error) throw new Error(evt.error);
                    if (evt.token) onToken(evt.token);
                    if (evt.done) {
                        doneData.tokens = evt.tokens;
                        doneData.elapsed = evt.elapsed;
                        return doneData;
                    }
                }
                return pump();
            });
        }
        return pump();
    }

    // 流结束（正常完成或用户停止）后的收尾
    function finishStream(assistant, cursor, info) {
        cursor.remove();
        var bubble = assistant.bubble;
        var meta = assistant.meta;
        var answer = bubble.textContent;

        history.push({ role: 'assistant', content: answer });

        // 统计信息：用时 X.XXs · N tokens（停止时 token 数按已接收文本长度估算）
        var parts = [];
        parts.push('用时 ' + (info.elapsed !== null ? Number(info.elapsed).toFixed(2) : '--') + 's');
        parts.push((info.tokens !== null ? info.tokens : answer.length) + ' tokens');
        if (info.stopped) parts.push('已停止');
        meta.textContent = parts.join(' · ');
        meta.classList.remove('hidden');

        setStreaming(false);
        abortController = null;
        scrollToBottom();
    }

    // ---------------- 事件绑定 ----------------
    // 发送按钮
    sendBtn.addEventListener('click', function () {
        sendMessage();
    });

    // 停止按钮：中断 fetch
    stopBtn.addEventListener('click', function () {
        if (abortController) abortController.abort();
    });

    // 清空对话
    clearBtn.addEventListener('click', function () {
        if (streaming) return;
        history = [];
        chatArea.querySelectorAll('.message-row').forEach(function (row) {
            row.remove();
        });
        welcome.classList.remove('hidden');
        messageInput.focus();
    });

    // 回车发送，Shift+回车换行
    messageInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    messageInput.addEventListener('input', autoGrow);

    // 示例问题：点击直接发送
    exampleChips.forEach(function (chip) {
        chip.addEventListener('click', function () {
            sendMessage(chip.textContent.trim());
        });
    });

    // 微调前 / 微调后 切换（仅影响下一次请求）
    modelBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            if (streaming) return;
            modelBtns.forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            currentModel = btn.getAttribute('data-model');
        });
    });

    // ---------------- 初始化 ----------------
    checkHealth();
    setInterval(checkHealth, 10000); // 每 10 秒刷新一次服务状态
    messageInput.focus();
})();
