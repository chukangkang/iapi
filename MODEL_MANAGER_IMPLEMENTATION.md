# 模型管理器实现总结

## 实现目标
解决Qwen Image和Qwen Edit模型切换导致OOM的问题，实现模型在CPU/GPU间的动态切换。

## 核心组件

### 1. ModelManager类 (`app/qwen_image_service.py`)
全局单例模型管理器，支持：
- 模型注册与注销
- 模型在CPU/GPU间动态切换
- 内存使用监控
- 自动卸载不活跃模型到CPU

**关键方法：**
- `register_model(name, pipe, size_mb)`: 注册模型
- `activate_model(name)`: 激活模型到GPU
- `unregister_model(name)`: 卸载模型到CPU
- `get_memory_info()`: 获取内存使用信息

### 2. QwenImageService集成
- 使用全局`_model_manager`实例
- `_get_pipeline()`方法支持模型切换检测
- 自动注册和激活模型

### 3. QwenEditService集成
- 导入并使用全局`_model_manager`实例
- `_get_pipeline()`方法支持模型切换检测
- 添加`_get_model_name()`, `_estimate_model_size()`, `_unload_pipeline()`方法

## 内存估算

### Qwen-Image-2512
- 4bit量化: ~4GB CPU内存
- FP16: ~14-16GB CPU内存

### Qwen-Image-Edit-2511
- 4bit量化: ~4GB CPU内存
- FP16: ~14-16GB CPU内存

### 总计
- 两个4bit模型: ~8-10GB CPU内存 ✓
- 两个FP16模型: ~110GB CPU内存 ✗ (55GB不足)

## 模型切换流程

1. **检测切换**: 比较当前模型名称与配置中的模型名称
2. **卸载旧模型**: 将当前模型移到CPU或完全卸载
3. **加载新模型**: 从CPU加载到GPU或从磁盘加载
4. **注册管理器**: 将模型注册到ModelManager
5. **激活模型**: 将模型移到GPU

## 日志输出

### 模型切换日志
```
Model switch detected: qwen_image_2512 -> qwen_image_edit_2511
Unloaded pipeline to CPU: qwen_image_2512
Registered model 'qwen_image_edit_2511': 4000.0 MB
Activated model 'qwen_image_edit_2511' on GPU
```

### 内存监控日志
```
Worker <name>/<id> task <id> start: GPU used=1024.0/8192.0 MB (12.5%), util=15%
Worker <name>/<id> task <id> end: GPU used=2048.0/8192.0 MB (25.0%), util=25%
```

## 使用建议

### 推荐配置（55GB CPU内存）
```python
# 使用4bit量化
QWEN_IMAGE_QUANTIZATION = "4bit"
QWEN_EDIT_QUANTIZATION = "4bit"
TORCH_DTYPE = "auto"  # 或不设置

# 模型路径
QWEN_IMAGE_MODEL_PATH = "/path/to/Qwen-Image-2512"
QWEN_EDIT_MODEL_PATH = "/path/to/Qwen-Image-Edit-2511"
```

### 避免的配置
```python
# 不要同时加载两个FP16模型
TORCH_DTYPE = "float16"  # 会导致OOM
```

## 异常处理

### OOM检测
- 子进程隔离，OOM不会影响主进程
- 完整日志记录GPU显存和进程状态
- 自动退出并记录堆栈跟踪

### 信号处理
- SIGSEGV: 段错误
- SIGBUS: 总线错误
- SIGFPE: 浮点异常
- SIGABRT: abort()调用

## 文件修改清单

1. **app/qwen_image_service.py**
   - 添加ModelManager类
   - 添加全局_model_manager实例
   - 修改QwenImageService._get_pipeline()

2. **app/qwen_edit_service.py**
   - 导入ModelManager和_model_manager
   - 添加logger
   - 修改__init__()添加_model_manager引用
   - 修改_get_pipeline()支持模型切换
   - 添加_get_model_name(), _estimate_model_size(), _unload_pipeline()

3. **app/worker.py**
   - 添加全局信号处理器
   - 添加MemoryError捕获
   - 添加GPU显存日志

4. **app/tasks.py**
   - 添加check_gpu_memory()函数
   - 添加check_process_health()函数
   - 添加_log_task_health()方法
   - 修改_run_task()添加健康检查

5. **requirements.txt**
   - 添加psutil>=6.0.0

6. **requirements-worker.txt**
   - 添加psutil>=6.0.0

## 测试建议

1. **模型切换测试**
   - 切换Qwen Image和Qwen Edit模型
   - 验证日志输出
   - 检查GPU显存使用

2. **OOM测试**
   - 尝试加载两个FP16模型
   - 验证OOM检测和日志
   - 验证子进程隔离

3. **内存监控测试**
   - 运行多个任务
   - 验证GPU显存监控
   - 验证进程健康检查

## 后续优化

1. **自动卸载策略**
   - 基于LRU自动卸载不活跃模型
   - 基于内存压力自动卸载

2. **模型预热**
   - 启动时预热常用模型
   - 减少首次请求延迟

3. **内存预警**
   - GPU显存超过阈值时预警
   - 自动触发模型卸载
