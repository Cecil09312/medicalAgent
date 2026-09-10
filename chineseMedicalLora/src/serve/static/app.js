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


/* ============================================================
 * 评估功能：视图切换 + 数据质量评估 + BLEU 模型评估
 *
 * 后端接口契约（FastAPI）：
 *   GET  /api/evaluate/data/files          -> { files: [{path, size_kb}] }
 *   POST /api/evaluate/data                <- { file_path, sample_size? }
 *   POST /api/evaluate/bleu                <- { pairs, use_inference, model }
 *   GET  /api/evaluate/history             -> { reports: [...] }
 *   GET  /api/evaluate/report/{filename}   -> 报告 JSON
 * ============================================================ */

(function () {
    'use strict';

    // ---------------- DOM 元素 ----------------
    var navTabs = document.querySelectorAll('.nav-tab');
    var views = document.querySelectorAll('.view');

    var dataFileSelect = document.getElementById('dataFileSelect');
    var dataSampleSize = document.getElementById('dataSampleSize');
    var dataEvalBtn = document.getElementById('dataEvalBtn');
    var dataHistoryBtn = document.getElementById('dataHistoryBtn');
    var dataHistoryPanel = document.getElementById('dataHistoryPanel');
    var dataHistoryList = document.getElementById('dataHistoryList');
    var dataEvalLoading = document.getElementById('dataEvalLoading');
    var dataEvalError = document.getElementById('dataEvalError');
    var dataEvalResult = document.getElementById('dataEvalResult');
    var dataScoreCards = document.getElementById('dataScoreCards');
    var dataDetailTable = document.getElementById('dataDetailTable');

    var modeBtns = document.querySelectorAll('.mode-btn');
    var bleuModelConfig = document.getElementById('bleuModelConfig');
    var bleuModelSelect = document.getElementById('bleuModelSelect');
    var bleuModeTip = document.getElementById('bleuModeTip');
    var bleuPairsBox = document.getElementById('bleuPairsBox');
    var bleuAddPairBtn = document.getElementById('bleuAddPairBtn');
    var bleuEvalBtn = document.getElementById('bleuEvalBtn');
    var bleuHistoryBtn = document.getElementById('bleuHistoryBtn');
    var bleuHistoryPanel = document.getElementById('bleuHistoryPanel');
    var bleuHistoryList = document.getElementById('bleuHistoryList');
    var bleuEvalLoading = document.getElementById('bleuEvalLoading');
    var bleuEvalError = document.getElementById('bleuEvalError');
    var bleuEvalResult = document.getElementById('bleuEvalResult');
    var bleuScoreCards = document.getElementById('bleuScoreCards');
    var bleuDetailTable = document.getElementById('bleuDetailTable');

    var charts = {};          // id -> echarts 实例
    var currentMode = 'manual';

    var PRIMARY = '#0d9488';
    var COLORS = ['#0d9488', '#0ea5e9', '#f59e0b', '#8b5cf6', '#ef4444', '#10b981', '#f97316', '#6366f1'];

    // ---------------- 视图切换 ----------------
    function switchView(name) {
        navTabs.forEach(function (tab) {
            tab.classList.toggle('active', tab.getAttribute('data-view') === name);
        });
        views.forEach(function (view) {
            view.classList.toggle('active', view.id === 'view-' + name);
        });
        // 切换后重绘当前可见视图的图表（display:none 下初始化的图表宽高为 0）
        setTimeout(resizeVisibleCharts, 50);
    }

    navTabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            switchView(tab.getAttribute('data-view'));
        });
    });

    function resizeVisibleCharts() {
        Object.keys(charts).forEach(function (id) {
            var el = document.getElementById(id);
            if (el && el.offsetWidth > 0 && charts[id]) charts[id].resize();
        });
    }
    window.addEventListener('resize', resizeVisibleCharts);

    // ---------------- 图表工具 ----------------
    function renderChart(id, option) {
        var el = document.getElementById(id);
        if (!el || typeof echarts === 'undefined') return;
        if (!charts[id] || charts[id].isDisposed()) charts[id] = echarts.init(el);
        charts[id].setOption(option, true);
        charts[id].resize();
    }

    function show(el, on) { el.classList.toggle('hidden', !on); }

    function showError(el, message) {
        el.textContent = message;
        show(el, true);
    }

    function postJSON(url, body) {
        return fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }).then(function (resp) {
            if (!resp.ok) {
                return resp.json().catch(function () { return {}; }).then(function (data) {
                    throw new Error(data.detail || 'HTTP ' + resp.status);
                });
            }
            return resp.json();
        });
    }

    // ---------------- 数据质量评估 ----------------
    function loadDataFiles() {
        fetch('/api/evaluate/data/files')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                dataFileSelect.innerHTML = '';
                (data.files || []).forEach(function (f) {
                    var opt = document.createElement('option');
                    opt.value = f.path;
                    opt.textContent = f.path + ' (' + f.size_kb + ' KB)';
                    dataFileSelect.appendChild(opt);
                });
                if (!dataFileSelect.options.length) {
                    dataFileSelect.innerHTML = '<option value="">未找到数据文件</option>';
                }
            })
            .catch(function () {
                dataFileSelect.innerHTML = '<option value="">加载失败</option>';
            });
    }

    function scoreCard(label, value, sub) {
        var card = document.createElement('div');
        card.className = 'score-card';
        var v = document.createElement('div');
        v.className = 'score-value';
        v.textContent = value;
        var l = document.createElement('div');
        l.className = 'score-label';
        l.textContent = label;
        card.appendChild(v);
        card.appendChild(l);
        if (sub) {
            var s = document.createElement('div');
            s.className = 'score-sub';
            s.textContent = sub;
            card.appendChild(s);
        }
        return card;
    }

    function renderDataReport(report) {
        var meta = report.meta || {};
        var score = report.quality_score || {};

        // 评分卡片
        dataScoreCards.innerHTML = '';
        dataScoreCards.appendChild(scoreCard('综合评分', score.total_score + '/100', score.grade || ''));
        dataScoreCards.appendChild(scoreCard('样本数', meta.sampled || meta.total_samples || 0));
        dataScoreCards.appendChild(scoreCard('格式合规率', pct(report.format_compliance && report.format_compliance.compliance_rate)));
        dataScoreCards.appendChild(scoreCard('中文占比', pct(report.language_consistency && report.language_consistency.chinese_ratio)));
        dataScoreCards.appendChild(scoreCard('问题重复率', pct(report.question_duplicates && report.question_duplicates.duplicate_rate)));
        dataScoreCards.appendChild(scoreCard('类别数', (report.category_distribution || {}).total_categories || 0));

        // 维度评分柱状图
        renderChart('chartDataScores', {
            color: [PRIMARY],
            tooltip: {},
            grid: { left: 90, right: 20, top: 10, bottom: 30 },
            xAxis: { type: 'value', max: 20 },
            yAxis: {
                type: 'category',
                data: [
                    ['多样性', 'diversity_score'], ['长度合理', 'length_score'], ['数据唯一', 'uniqueness_score'],
                    ['语言一致', 'language_score'], ['字段完整', 'completeness_score'], ['格式合规', 'format_score']
                ].map(function (x) { return x[0]; })
            },
            series: [{
                type: 'bar',
                barWidth: 18,
                label: { show: true, position: 'right' },
                data: [
                    score.diversity_score, score.length_score, score.uniqueness_score,
                    score.language_score, score.completeness_score, score.format_score
                ]
            }]
        });

        // 类别分布饼图
        var dist = (report.category_distribution || {}).distribution || {};
        var distData = Object.keys(dist).slice(0, 12).map(function (k) {
            return { name: k, value: dist[k].count };
        });
        renderChart('chartCategory', {
            color: COLORS,
            tooltip: { trigger: 'item', formatter: '{b}: {c} 条 ({d}%)' },
            legend: { type: 'scroll', orient: 'vertical', right: 0, top: 'middle', textStyle: { fontSize: 11 } },
            series: [{
                type: 'pie',
                radius: ['35%', '65%'],
                center: ['35%', '50%'],
                label: { show: false },
                data: distData.length ? distData : [{ name: '无类别字段', value: 0 }]
            }]
        });

        // 长度统计柱状图
        var q = report.question_length || {};
        var a = report.answer_length || {};
        var t = report.token_estimation || {};
        renderChart('chartLength', {
            color: COLORS,
            tooltip: { trigger: 'axis' },
            legend: { top: 0 },
            grid: { left: 50, right: 20, top: 34, bottom: 30 },
            xAxis: { type: 'category', data: ['最小', 'P10', '中位数', '平均', 'P90', '最大'] },
            yAxis: { type: 'value', name: '字符数' },
            series: [
                { name: '问题长度', type: 'bar', data: [q.min_length, q.p10, q.median_length, q.mean_length, q.p90, q.max_length] },
                { name: '回答长度', type: 'bar', data: [a.min_length, a.p10, a.median_length, a.mean_length, a.p90, a.max_length] }
            ]
        });

        // 语言分布饼图
        var lang = report.language_consistency || {};
        renderChart('chartLanguage', {
            color: [PRIMARY, '#0ea5e9', '#f59e0b'],
            tooltip: { trigger: 'item', formatter: '{b}: {c} 条 ({d}%)' },
            legend: { bottom: 0 },
            series: [{
                type: 'pie',
                radius: ['35%', '62%'],
                label: { show: false },
                data: [
                    { name: '中文为主', value: lang.chinese_dominant || 0 },
                    { name: '英文为主', value: lang.english_dominant || 0 },
                    { name: '中英混合', value: lang.mixed_language || 0 }
                ]
            }]
        });

        // 详细指标表
        var rows = [
            ['数据来源', meta.data_source || '-'],
            ['报告时间', meta.report_time || '-'],
            ['格式合规', (report.format_compliance || {}).compliant + ' / ' + (report.format_compliance || {}).total + ' 条 (' + pct((report.format_compliance || {}).compliance_rate) + ')'],
            ['问题长度', '平均 ' + (q.mean_length || 0) + ' 字，极短(<10字) ' + (q.very_short_count || 0) + ' 条，极长(>1000字) ' + (q.very_long_count || 0) + ' 条'],
            ['回答长度', '平均 ' + (a.mean_length || 0) + ' 字，极短 ' + (a.very_short_count || 0) + ' 条，极长 ' + (a.very_long_count || 0) + ' 条'],
            ['Token 估算', '平均 ' + (t.mean_tokens || 0) + ' tokens，>512: ' + (t.over_512 || 0) + ' 条，>1024: ' + (t.over_1024 || 0) + ' 条，>2048: ' + (t.over_2048 || 0) + ' 条'],
            ['问题重复', (report.question_duplicates || {}).duplicates + ' 条重复 (' + pct((report.question_duplicates || {}).duplicate_rate) + ')'],
            ['回答重复', (report.answer_duplicates || {}).duplicates + ' 条重复 (' + pct((report.answer_duplicates || {}).duplicate_rate) + ')']
        ];
        // 字段完整性
        var fc = report.field_completeness || {};
        Object.keys(fc).forEach(function (field) {
            rows.push(['字段完整 · ' + field, fc[field].filled + ' / ' + (fc[field].filled + fc[field].empty) + ' 条 (' + pct(fc[field].fill_rate) + ')']);
        });
        // 格式问题
        var issues = (report.format_compliance || {}).issues || {};
        Object.keys(issues).forEach(function (k) {
            rows.push(['格式问题 · ' + k, issues[k] + ' 条']);
        });

        var html = '<thead><tr><th style="width:200px">指标</th><th>结果</th></tr></thead><tbody>';
        rows.forEach(function (r) {
            html += '<tr><td class="metric-name">' + esc(r[0]) + '</td><td>' + esc(r[1]) + '</td></tr>';
        });
        html += '</tbody>';
        dataDetailTable.innerHTML = html;

        show(dataEvalResult, true);
        setTimeout(resizeVisibleCharts, 60);
    }

    function pct(v) {
        return (v === undefined || v === null) ? '-' : (v * 100).toFixed(2) + '%';
    }

    dataEvalBtn.addEventListener('click', function () {
        var filePath = dataFileSelect.value;
        if (!filePath) { showError(dataEvalError, '请先选择数据文件'); return; }
        show(dataEvalError, false);
        show(dataEvalResult, false);
        show(dataHistoryPanel, false);
        show(dataEvalLoading, true);
        dataEvalBtn.disabled = true;

        var body = { file_path: filePath };
        var sampleSize = parseInt(dataSampleSize.value, 10);
        if (sampleSize >= 10) body.sample_size = sampleSize;

        postJSON('/api/evaluate/data', body)
            .then(function (report) {
                show(dataEvalLoading, false);
                dataEvalBtn.disabled = false;
                renderDataReport(report);
            })
            .catch(function (err) {
                show(dataEvalLoading, false);
                dataEvalBtn.disabled = false;
                showError(dataEvalError, '评估失败：' + err.message);
            });
    });

    // ---------------- BLEU 模型评估 ----------------
    function createPairRow(question, reference, candidate) {
        var row = document.createElement('div');
        row.className = 'pair-row';

        var head = document.createElement('div');
        head.className = 'pair-head';
        var title = document.createElement('span');
        title.className = 'pair-title';
        head.appendChild(title);
        var del = document.createElement('button');
        del.type = 'button';
        del.className = 'pair-del';
        del.textContent = '删除';
        del.addEventListener('click', function () {
            row.remove();
            renumberPairs();
        });
        head.appendChild(del);
        row.appendChild(head);

        function addField(labelText, value, placeholder) {
            var field = document.createElement('div');
            field.className = 'pair-field';
            var label = document.createElement('label');
            label.textContent = labelText;
            var textarea = document.createElement('textarea');
            textarea.rows = 2;
            textarea.placeholder = placeholder;
            textarea.value = value || '';
            field.appendChild(label);
            field.appendChild(textarea);
            row.appendChild(field);
            return textarea;
        }

        var qInput = addField('问题', question, currentMode === 'inference' ? '问题（用于生成模型回答）' : '问题（可选）');
        var rInput = addField('参考答案', reference, '标准答案');
        var cInput = addField('模型回答', candidate, '模型生成的回答');
        if (currentMode === 'inference') {
            cInput.parentElement.classList.add('hidden');
        }

        row._inputs = { q: qInput, r: rInput, c: cInput };
        return row;
    }

    function renumberPairs() {
        var rows = bleuPairsBox.querySelectorAll('.pair-row');
        rows.forEach(function (row, i) {
            row.querySelector('.pair-title').textContent = '样本 ' + (i + 1);
        });
    }

    bleuAddPairBtn.addEventListener('click', function () {
        bleuPairsBox.appendChild(createPairRow('', '', ''));
        renumberPairs();
    });

    // 在标红的输入框里输入内容后立即取消高亮
    bleuPairsBox.addEventListener('input', function (e) {
        if (e.target && e.target.classList) e.target.classList.remove('field-missing');
    });

    modeBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            currentMode = btn.getAttribute('data-mode');
            modeBtns.forEach(function (b) { b.classList.toggle('active', b === btn); });
            show(bleuModelConfig, currentMode === 'inference');
            bleuModeTip.textContent = currentMode === 'inference'
                ? '输入问题和参考答案，系统调用所选模型生成回答后自动计算 BLEU 分数（首次调用需加载模型，耗时较长）。'
                : '粘贴或输入「参考答案 + 模型回答」文本对，系统将计算 BLEU 分数评估模型输出质量。';
            // 切换每行「模型回答」输入框的显隐
            bleuPairsBox.querySelectorAll('.pair-row').forEach(function (row) {
                var cField = row._inputs.c.parentElement;
                if (currentMode === 'inference') { cField.classList.add('hidden'); }
                else { cField.classList.remove('hidden'); }
            });
        });
    });

    function collectPairs() {
        var pairs = [];
        bleuPairsBox.querySelectorAll('.pair-row').forEach(function (row) {
            var q = row._inputs.q.value.trim();
            var r = row._inputs.r.value.trim();
            var c = row._inputs.c.value.trim();
            if (r) pairs.push({ question: q, reference: r, candidate: c });
        });
        return pairs;
    }

    // 本地校验：缺失字段标红并提示，返回 null 表示校验失败
    function validatePairs() {
        var firstMissing = null;
        var missingCount = 0;
        bleuPairsBox.querySelectorAll('.pair-row').forEach(function (row, idx) {
            var fields = [
                { input: row._inputs.r, name: '参考答案' },
                { input: row._inputs.c, name: '模型回答', skip: currentMode === 'inference' }
            ];
            fields.forEach(function (f) {
                if (!f.skip && !f.input.value.trim()) {
                    f.input.classList.add('field-missing');
                    f.input.placeholder = f.name + '不能为空，请填写';
                    if (!firstMissing) {
                        firstMissing = { idx: idx + 1, name: f.name };
                    }
                    missingCount++;
                }
            });
        });
        if (missingCount > 0) {
            showError(bleuEvalError, '第 ' + firstMissing.idx + ' 组样本缺少「' + firstMissing.name +
                '」（共 ' + missingCount + ' 处未填写，已用红框标出）。' +
                (currentMode === 'inference' ? '推理模式下只需填写问题和参考答案。' : '手动模式下「参考答案」和「模型回答」均为必填。'));
            return null;
        }
        show(bleuEvalError, false);
        return true;
    }

    function renderBleuReport(report) {
        var s = report.summary || {};

        bleuScoreCards.innerHTML = '';
        bleuScoreCards.appendChild(scoreCard('平均 BLEU', (s.average_bleu || 0).toFixed(4)));
        bleuScoreCards.appendChild(scoreCard('中位数', (s.median_bleu || 0).toFixed(4)));
        bleuScoreCards.appendChild(scoreCard('最高 / 最低', (s.max_bleu || 0).toFixed(2) + ' / ' + (s.min_bleu || 0).toFixed(2)));
        bleuScoreCards.appendChild(scoreCard('样本数', s.total_pairs || 0));
        bleuScoreCards.appendChild(scoreCard('优质 (≥0.5)', (s.good_count || 0) + ' 条'));
        bleuScoreCards.appendChild(scoreCard('较差 (<0.2)', (s.poor_count || 0) + ' 条'));

        // n-gram 精度（取所有样本平均）
        var details = report.details || [];
        var maxN = 4;
        var avgPrecisions = [];
        for (var n = 0; n < maxN; n++) {
            var sum = 0, cnt = 0;
            details.forEach(function (d) {
                var p = d.precisions || [];
                if (p[n] !== undefined) { sum += p[n]; cnt++; }
            });
            avgPrecisions.push(cnt ? +(sum / cnt).toFixed(4) : 0);
        }
        renderChart('chartBleuNgram', {
            color: [PRIMARY],
            tooltip: { trigger: 'axis' },
            grid: { left: 50, right: 20, top: 20, bottom: 30 },
            xAxis: { type: 'category', data: ['1-gram', '2-gram', '3-gram', '4-gram'] },
            yAxis: { type: 'value', max: 1 },
            series: [{
                name: '平均精度', type: 'bar', barWidth: 40,
                label: { show: true, position: 'top' },
                data: avgPrecisions
            }]
        });

        // 逐条样本分数
        renderChart('chartBleuPerPair', {
            color: COLORS,
            tooltip: { trigger: 'axis' },
            grid: { left: 50, right: 20, top: 20, bottom: 40 },
            xAxis: { type: 'category', data: details.map(function (d, i) { return '#' + (i + 1); }) },
            yAxis: { type: 'value', max: 1 },
            series: [{
                name: 'BLEU', type: 'bar',
                label: { show: true, position: 'top', formatter: function (p) { return p.value.toFixed(2); } },
                data: details.map(function (d) { return d.bleu_score || 0; })
            }]
        });

        // 明细表
        var html = '<thead><tr><th style="width:36px">#</th><th style="width:70px">BLEU</th><th>问题</th><th>参考答案</th><th>模型回答</th></tr></thead><tbody>';
        details.forEach(function (d, i) {
            html += '<tr>'
                + '<td>' + (i + 1) + '</td>'
                + '<td class="' + (d.bleu_score >= 0.5 ? 'score-good' : d.bleu_score < 0.2 ? 'score-poor' : '') + '">' + (d.bleu_score || 0).toFixed(4) + '</td>'
                + '<td class="cell-text">' + esc(d.question || d.reference) + '</td>'
                + '<td class="cell-text">' + esc(d.reference_full || d.reference) + '</td>'
                + '<td class="cell-text">' + esc(d.candidate_full || d.candidate) + '</td>'
                + '</tr>';
        });
        html += '</tbody>';
        bleuDetailTable.innerHTML = html;

        show(bleuEvalResult, true);
        setTimeout(resizeVisibleCharts, 60);
    }

    bleuEvalBtn.addEventListener('click', function () {
        // 本地校验：缺失字段标红提示，不再发请求到后端
        if (validatePairs() === null) return;
        var pairs = collectPairs();
        if (!pairs.length) { showError(bleuEvalError, '请至少填写一组有效的「参考答案」'); return; }
        show(bleuEvalError, false);
        show(bleuEvalResult, false);
        show(bleuHistoryPanel, false);
        show(bleuEvalLoading, true);
        bleuEvalBtn.disabled = true;

        postJSON('/api/evaluate/bleu', {
            pairs: pairs,
            use_inference: currentMode === 'inference',
            model: bleuModelSelect.value
        })
            .then(function (report) {
                show(bleuEvalLoading, false);
                bleuEvalBtn.disabled = false;
                renderBleuReport(report);
            })
            .catch(function (err) {
                show(bleuEvalLoading, false);
                bleuEvalBtn.disabled = false;
                showError(bleuEvalError, '评估失败：' + err.message);
            });
    });

    // ---------------- 历史报告 ----------------
    function toggleHistory(panel, list, renderFn) {
        if (!panel.classList.contains('hidden')) { show(panel, false); return; }
        fetch('/api/evaluate/history')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                renderHistoryList(list, data.reports || [], panel, renderFn);
                show(panel, true);
            })
            .catch(function () { showError(dataEvalError, '历史加载失败'); });
    }

    function renderHistoryList(list, reports, panel, renderFn) {
        list.innerHTML = '';
        if (!reports.length) {
            list.innerHTML = '<p class="history-empty">暂无历史报告</p>';
            return;
        }
        reports.forEach(function (r) {
            var item = document.createElement('button');
            item.type = 'button';
            item.className = 'history-item';
            var left = document.createElement('div');
            left.className = 'history-info';
            left.innerHTML = '<span class="history-title">' + esc(r.title || r.file) + '</span>'
                + '<span class="history-meta">' + (r.kind === 'data_quality' ? '数据质量' : 'BLEU') + ' · ' + esc(r.grade || '') + '</span>';
            var score = document.createElement('span');
            score.className = 'history-score';
            score.textContent = (r.score === undefined || r.score === null) ? '-' : r.score;
            item.appendChild(left);
            item.appendChild(score);
            item.addEventListener('click', function () {
                fetch('/api/evaluate/report/' + encodeURIComponent(r.file))
                    .then(function (resp) {
                        if (!resp.ok) throw new Error('HTTP ' + resp.status);
                        return resp.json();
                    })
                    .then(function (report) {
                        show(panel, false);
                        renderFn(report);
                    })
                    .catch(function (e) { alert('加载历史报告失败：' + e.message); });
            });
            list.appendChild(item);
        });
    }

    dataHistoryBtn.addEventListener('click', function () {
        toggleHistory(dataHistoryPanel, dataHistoryList, renderDataReport);
    });
    bleuHistoryBtn.addEventListener('click', function () {
        toggleHistory(bleuHistoryPanel, bleuHistoryList, renderBleuReport);
    });

    function esc(text) {
        if (text === undefined || text === null) return '';
        return String(text).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    // ---------------- 初始化 ----------------
    loadDataFiles();
    bleuPairsBox.appendChild(createPairRow('', '', ''));
    renumberPairs();
})();
