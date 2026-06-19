# Worker增强功能说明

## 问题分析
原Worker存在以下问题:
1. **缺少全局异常捕获** - OOM或致命信号会导致进程直接退出,无日志
2. **缺少进程监控** - 无法检测进程是否存活
3. **缺少GPU显存监控** - 无法在OOM前预警
4. **缺少子进程隔离** - 主进程可能被子任务的崩溃影响

## 实现的功能

### 1. 全局异常捕获 (`app/worker.py`)
- 捕获`MemoryError` (OOM) 并记录详细日志
- 捕获致命信号 (SIGSEGV, SIGBUS, SIGFPE, SIGABRT)
- 记录GPU显存信息和进程状态
- 使用`os._exit()`确保进程立即退出

### 2. 子进程隔离 (`app/tasks.py`)
- 所有任务在子进程中执行
- 主进程监控子进程状态
- 子进程OOM不会影响主进程

### 3. 健康检查 (`app/tasks.py`)
- 任务开始前/后检查GPU显存
- 检查进程健康状态 (PID, 内存, 线程数等)
- 记录详细日志用于调试

### 4. 信号处理 (`app/worker.py`)
```python
SIGSEGV: 段错误(内存访问违规)
SIGBUS:  总线错误(内存对齐问题)  
SIGFPE:  浮点异常(可能由OOM触发)
SIGABRT: abort()调用
```

## 新增依赖
- `psutil>=6.0.0` - 进程和系统监控

## 日志输出示例

### 正常任务
```
2026-06-19 17:46:16,224 INFO [app.tasks] Worker node-2/0 started task img-aeb3969888134a34b06a51365e63503e
2026-06-19 17:46:16,225 INFO [app.tasks] Task details: id=img-xxx, width=1104, height=1472, steps=2
2026-06-19 17:46:16,226 INFO [app.tasks] Worker node-2/0 task img-xxx start: GPU used=1024.0/8192.0 MB (12.5%), util=15%
2026-06-19 17:46:55,547 INFO [app.tasks] Worker node-2/0 completed task img-aeb3969888134a34b06a51365e63503e
2026-06-19 17:46:55,548 INFO [app.tasks] Worker node-2/0 task img-xxx end: GPU used=2048.0/8192.0 MB (25.0%), util=0%
```

### OOM场景
```
2026-06-19 17:46:16,224 INFO [app.tasks] Worker node-2/0 started task img-xxx
2026-06-19 17:46:16,225 INFO [app.tasks] Task details: id=img-xxx, width=4096, height=4096, steps=50
2026-06-19 17:46:16,226 INFO [app.tasks] Worker node-2/0 task img-xxx start: GPU used=6000.0/8192.0 MB (73.2%), util=85%
2026-06-19 17:47:24,366 CRITICAL [app.tasks] Worker 0/node-2 OOM on task img-xxx
2026-06-19 17:47:24,367 CRITICAL [app.tasks] Task payload: {...}
2026-06-19 17:47:24,368 CRITICAL [app.tasks] Worker PID: 12345
2026-06-19 17:47:24,369 CRITICAL [app.tasks] GPU memory info: 7500,8192,95
2026-06-19 17:47:24,370 CRITICAL [app.worker] MemoryError (OOM) detected: CUDA out of memory
2026-06-19 17:47:24,371 CRITICAL [app.worker] Worker process PID: 12345
2026-06-19 17:47:24,372 CRITICAL [app.worker] Worker process PPID: 12340
```

### 进程崩溃
```
2026-06-19 17:46:16,224 INFO [app.tasks] Worker node-2/0 started task img-xxx
2026-06-19 17:46:41,386 CRITICAL [app.worker] Received SIGSEGV (signal 11), worker is terminating
2026-06-19 17:46:41,387 CRITICAL [app.worker] Stack trace:
  File "app/worker.py", line 123, in main
  File "app/tasks.py", line 456, in _run_task
2026-06-19 17:46:41,388 CRITICAL [app.worker] Worker process PID: 12345
2026-06-19 17:46:41,389 CRITICAL [app.worker] Worker process PPID: 12340
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements-worker.txt
```

### 启动Worker
```bash
python -m app.worker
```

### 监控建议
1. **监控日志文件** - 关注`CRITICAL`级别的日志
2. **监控GPU显存** - 使用`nvidia-smi -l 1`观察显存使用趋势
3. **监控进程状态** - 使用`ps aux | grep worker`检查进程是否存活
4. **设置告警** - 当GPU使用率超过80%时告警

## 注意事项
1. 子进程隔离会增加少量启动开销 (~1-2秒)
2. 建议在生产环境使用`SERVICE_ROLE=worker`模式
3. 定期检查日志,及时发现潜在OOM问题
4. 如果频繁OOM,考虑:
   - 减小`DEFAULT_WIDTH`和`DEFAULT_HEIGHT`
   - 减小`NUM_INFERENCE_STEPS`
   - 增加GPU显存
