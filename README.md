# 车间质检台 · ShopInspect

面向产线工位的外观质检 **应用台**（V1.1）：摄像头/图片检测 → YOLO 推理 → FastAPI → SQLite 追溯 → Web 看板。

> 定位：**AI 应用工程师** 作品。先跑通闭环与可演示，不追求自研缺陷 mAP。V1 使用官方通用 YOLO 权重验证通路；缺陷专用模型见 V2。

## 能力（V1）

| 能力 | 说明 |
|------|------|
| 摄像头实时检测 | `scripts/run_cam.py`（`q` 退出，`s` 保存并落库） |
| 图片/路径检测 | API `POST /detect/image`、`POST /detect/path` |
| 结构化结果 | label / confidence / bbox_xyxy |
| 历史追溯 | SQLite `data/shopinspect.db` + `GET /records` |
| Web 看板 | http://127.0.0.1:8787/ 上传检测 / 可选摄像头 / 查记录 |
| 可切换权重 | `config.yaml` → `model_path`（预留 `models/defect_best.pt`） |

## 快速开始（Windows / CPU）

```powershell
cd A:\AI视觉\ShopInspect

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CPU 版 torch（若默认源慢，可换清华等镜像）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 探测摄像头
python scripts/probe_camera.py

# 实时窗口（需本机有界面）
python scripts/run_cam.py

# API + 看板
python scripts/run_api.py
# 浏览器打开 http://127.0.0.1:8787/
# Swagger: http://127.0.0.1:8787/docs
```

### 摄像头注意

1. Windows 设置 → 隐私和安全性 → 相机 → 允许桌面应用访问
2. 关掉 Teams / 微信 /「相机」应用占用
3. 默认 index=`0`（可在 `config.yaml` 改 `camera_index`）



## V1.1 优化

- 推理前长边缩放（`max_infer_side`，默认 960）加速大图/摄像头
- JPEG 质量可配（`jpeg_quality`）
- API 返回 `elapsed_ms` / `conf_used`；支持表单 `conf`
- `GET /stats` 统计；`DELETE /records/{id}` 删除
- 看板：置信度滑条、来源筛选、删除记录、类别 chips
- 连续检测默认**不落库**（可勾选落库），减少磁盘写入
- 看板 HTML `Cache-Control: no-store`，改 UI 刷新即生效

## 网页摄像头

看板顶部可切换 **上传图片** / **使用摄像头**：

1. 点「使用摄像头」→「开启摄像头」（浏览器授权）
2. **拍一帧检测**：抓当前画面送 YOLO，结果落库（source=`camera`）
3. **连续检测**：约 1.5 秒一帧（CPU 友好，可点停止）
4. 不需要摄像头时保持「上传图片」即可

说明：网页走浏览器 `getUserMedia`，与 `scripts/run_cam.py`（OpenCV 桌面窗）是两条通路，都可把结果写入同一数据库。

## 配置

见 `config.yaml`：

- `model_path`: 默认 `yolo11n.pt`（首次自动下载；也可放到 `models/`）
- `confidence` / `iou` / `device`
- `host` / `port`

## API 摘要

- `GET /health`
- `POST /detect/image` multipart: `file`, 可选 `note`, `return_annotated`, `save`
- `POST /detect/path` JSON: `{ "path": "...", "recursive": false }`
- `GET /records?limit=20&offset=0`
- `GET /records/{id}`
- `GET /files/{relative_path}` 访问 `data/outputs/...` 标注图

## 目录

```
ShopInspect/
  app/           # FastAPI + 检测/DB/摄像头
  scripts/       # probe / run_cam / run_api / smoke_test
  data/          # inputs outputs sqlite
  models/        # 本地权重（gitignore *.pt）
  config.yaml
```

## 简历一句话

独立完成车间质检台 ShopInspect：摄像头/图片 YOLO 检测、FastAPI 服务、SQLite 结果追溯与 Web 看板，打通产线视觉应用闭环（V1 通用模型验证通路，可切换自训缺陷权重）。

## V2 预留（未实现）

- 自训缺陷权重 `models/defect_best.pt`
- 批次号 / 工单号
- WebSocket 推流
- Java/Spring 或 MES 对接
- 缺陷 SOP 知识库问答

## 与 MES 对接（预留口径）

当前检测结果经 REST 落库，后续可由 Java 业务层消费 `/records` 或在检测成功回调中推送工单系统；V1 不实现 MES 协议本身。

## v0.1.3 增量

- 检测可填 **工单号 / 批次号**，落库可筛
- 历史支持 **类别 chips**、工单/批次筛选
- GET /records/export.csv 导出当前筛选结果（Excel 可直接开，UTF-8 BOM）

