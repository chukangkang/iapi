# 独立 SUPIR Worker 部署

> SUPIR 官方声明仅允许非商业使用。商业部署前必须取得作者书面许可。

## 硬件要求

- NVIDIA GPU；官方低显存方式仍约需 12GB 显存（不启用 LLaVA）
- 推荐 16GB 以上显存；完整 LLaVA 描述链路建议单独使用另一张 16GB GPU
- NVIDIA Container Toolkit 和 Docker Compose v2

当前开发机检测结果为 MX450 2GB 且未安装 Docker，因此不能在该机器执行真实 SUPIR 推理。

## 模型准备

将下列官方模型放入同一宿主机目录：

- `sd_xl_base_1.0_0.9vae.safetensors`
- `SUPIR-v0Q.ckpt`
- 可选 `SUPIR-v0F.ckpt`

SDXL 来自 `stabilityai/stable-diffusion-xl-base-1.0`。SUPIR Q/F 权重必须从官方 README 提供的 Google Drive 或百度网盘下载。不要使用来源不明的重新打包权重。

## 启动

1. 将 `.env.example` 复制为 `.env`，设置 `SUPIR_MODELS_DIR` 和随机 API Key。
2. 在本目录执行 `docker compose up --build -d`。
3. 等待 `/health` 返回 `{"ready": true, "model": "Q"}`。
4. 在 iapi Worker 中设置：
   - `SUPIR_ENABLED=true`
   - `SUPIR_BASE_URL=http://<SUPIR主机>:8010`
   - `SUPIR_API_KEY=<相同API Key>`

镜像固定官方提交 `bda91af2000042f8bedfec8897d92917e67c1d88`，避免上游变化破坏部署。

## 真实图片基准

在 iapi 仓库根目录运行 `scripts/benchmark_supir.py`，输入目录应包含真实的 jpg/jpeg/png/webp 图片。示例目录为 `/root/iapi/images`。输出包括：

- 每张恢复图片
- `records.csv`：延迟、输出像素、细节增益、相对输入的像素 MAE
- `summary.json`：成功率、平均/P50/P95 延迟、百万像素吞吐

像素 MAE 只是保真度代理指标，不代表感知质量。正式验收还应进行盲测和身份一致性检查。