---
title: MLOps & Observability
published: 2026-04-18
description: 系统讲解大模型推理服务的MLOps与可观测性体系构建：从Kubernetes + GPU基础设施层出发，剖析弹性伸缩三件套gistry的模型版本与Stage管理、DVC的数据版本控制；系统阐述可观测性三支柱在推理服务中的落地，并提炼出延迟/流量/错误/饱和度的四大黄金信号。
cover: "/assets/images/posts/mlops_observability.png"
coverInContent: false
tags: [MLOps, Kubernetes, GPU调度, 可观测性]
category: AI_Infra
draft: false
---

# MLOps & Observability

## 引言：大模型上线，不是“部署”而是“治理”

把一个大模型部署到生产环境，代码跑通只是开始。真正的挑战从上线那一刻才真正浮现：

- 早上8点流量高峰，GPU利用率冲到95%，但TTFT突然飙升到2秒——是调度问题还是资源不足？
- 训练了一个新版本模型，指标看着不错，上线后却效果下降——用的数据集是哪个版本？代码是哪个commit？
- 凌晨3点收到告警，P99延迟超了3倍——是哪一步慢了？Prefill还是Decode？哪个请求导致的？

这些问题单靠“部署”解决不了，需要的是**治理（Governance）** ——一套覆盖**代码→数据→模型→服务**全链路的MLOps体系。

本文从基础设施层（K8s + GPU调度）出发，穿过资产管理层（实验跟踪 + 模型注册 + 数据版本），抵达可观测性层（Metrics + Logs + Traces），构建一个完整的LLM推理服务MLOps全景图。

---

## 一、Kubernetes + GPU：基础设施的“地基”

### 1.1 GPU在K8s中从来不是“一等公民”

在Kubernetes的世界里，CPU和内存是“一等公民”：
- 可压缩/不可压缩资源
- Requests/Limits语义清晰
- CFS调度、QoS等级完善

而GPU呢？本质上是一个“**整块资源**”——不能被kube-scheduler原生切分，不能像CPU一样时间片抢占，不能像内存一样swap。所以Kubernetes一开始的态度是：“GPU？我不懂，你自己搞。”

于是 **Device Plugin** 登场了。

### 1.2 Device Plugin：GPU进入K8s的“通行证”

Device Plugin的核心作用只有一句话：**把物理设备翻译成K8s能听懂的资源**。

Kubernetes在Pod的API对象里并没有为GPU专门设置资源类型字段，而是使用了一种叫 **Extended Resource（ER）** 的特殊字段来传递GPU信息。

以NVIDIA GPU为例，整个流程是这样的：

**第一步：安装与运行**

`nvidia-device-plugin` 以 **DaemonSet** 形式运行在每个GPU节点上。它启动后会创建自己的Unix Socket文件（如`nvidia-gpu.sock`）并开始监听。

**第二步：向kubelet注册**

插件作为客户端向kubelet发送注册请求，内容包括：
- **设备名称**：`nvidia.com/gpu`——告诉kubelet这是什么资源
- **Unix Socket**：用于本地gRPC通信
- **API版本**：确保双方协议兼容

**第三步：设备发现与上报**

注册成功后，插件启动gRPC服务器，通过 `ListAndWatch` API与kubelet建立持久连接。kubelet持续监听设备ID及其健康状态，将设备信息整合到Node状态中，上报给API Server。

**第四步：调度器决策**

kube-scheduler查询API Server中的扩展资源信息，在调度时直接将可用量减去Pod声明的数值。

**第五步：Pod声明与分配**

用户只需在Pod的`limits`中声明GPU数量（注意：**GPU只能写在limits，不能写在requests，因为GPU不能被超卖**）：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-inference
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: 1  # 申请1张GPU
```

Pod创建时，kubelet调用Allocate接口，将GPU设备注入容器。

### 1.3 GPU共享：一张卡真的只能一个任务用吗？

整卡独占简单直接，但推理场景下往往太浪费——一个小模型推理任务占一整张A100，好比“杀鸡用牛刀”。NVIDIA提供了几种GPU共享方案：

**第一层：MIG（硬隔离，推荐）**

如果你用的是A100/H100，MIG（Multi-Instance GPU）可以将一张卡物理切分为多个独立实例。这是“真隔离”，不是时间片。在K8s中表现为：

```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1  # 申请1个1g.5gb的MIG实例
```

MIG是目前最“像CPU的GPU”。

**第二层：时间切片 / 共享（高级玩法，风险也高）**

CUDA MPS、GPU Time Slicing等技术可以实现GPU时间片共享，适合推理高并发场景。但代价也很明显：性能抖动、OOM、不可预测是家常便饭。

**第三层：NVIDIA GPU Operator（企业级推荐）**

对于生产环境，更推荐使用 **NVIDIA GPU Operator**——它自动处理驱动安装、MIG支持、完整监控，是“开箱即用”的企业级方案。

---

## 二、弹性伸缩：让GPU“按需而动”

推理服务的流量特征天然具有**潮汐性**——白天高、夜间低，工作日高、周末低。如果不能自动伸缩，要么在高峰期服务崩溃，要么在低谷期浪费GPU成本。

### 2.1 HPA：水平伸缩（Horizontal Pod Autoscaler）

HPA根据指标自动调整Pod副本数。对于推理服务，最直接的伸缩指标是 **QPS（每秒请求数）** 或 **GPU利用率**。

**基于QPS的HPA配置**（需要自定义metrics API）：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "10"  # 每个Pod平均10 QPS时触发扩容
```

### 2.2 VPA：垂直伸缩（Vertical Pod Autoscaler）

VPA根据实际负载自动调整Pod的CPU/内存资源请求。对于推理服务，VPA可以动态调整Pod的显存请求——但需要注意的是，**VPA调整需要重启Pod**，不适合对延迟敏感的服务。

**生产环境的最佳实践**：先以 **Off模式** 部署VPA，只获取推荐值不做实际变更；分析推荐值后手动调整，或对**内存**启用Auto模式（让VPA自动调整内存），CPU维度由HPA处理。

### 2.3 Cluster Autoscaler：节点级伸缩

当HPA/VPA触发了新的Pod调度，但集群中没有足够的GPU节点时，**Cluster Autoscaler** 会自动增加节点。三种伸缩器需要**协同配合**：

| 伸缩器 | 作用维度 | 伸缩对象 | 触发条件 |
|--------|---------|---------|---------|
| **HPA** | Pod副本数 | 水平扩展/收缩Pod | QPS/CPU/GPU利用率 |
| **VPA** | Pod资源配额 | 调整Pod的requests/limits | 历史负载分析 |
| **Cluster Autoscaler** | 节点数量 | 增加/删除Node | Pod因资源不足而Pending |

**注意**：HPA和VPA如果在同一个CPU维度上同时运行，会产生**冲突**。一般建议：HPA管CPU横向扩展，VPA管内存纵向调整。

---

## 三、实验跟踪与模型注册表：让“哪个模型好”有据可查

大模型迭代速度快，每周可能有多个版本。没有系统化的管理，团队很快就会陷入“哪个模型文件对应哪个实验”的混乱。

### 3.1 MLflow Model Registry：模型生命周期的“控制台”

MLflow Model Registry是MLflow平台的核心组件，提供了一个**中心化的模型存储库**，专门管理机器学习模型的完整生命周期。

**核心功能**：

1. **版本控制**：每个模型迭代自动进行版本控制和跟踪。可以比较不同迭代的性能指标、回滚到先前版本、维护完整的开发历史。

2. **Stage管理**：为模型版本定义标准化的生命周期阶段：
   - **Staging（预发布/测试）** ：验证中的模型
   - **Production（生产）** ：线上正在服务的模型
   - **Archived（归档）** ：已下线的历史版本

3. **模型血缘与可追溯性**：每个注册的模型版本都与产生它的MLflow Run、训练代码、数据集相关联——**完全可重现**。

**使用示例**：

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 注册模型
mlflow.register_model(
    "runs:/<run_id>/model",
    "llama-3.1-70b-instruct"
)

# 获取模型版本列表
model_versions = client.get_latest_versions("llama-3.1-70b-instruct")

# 转移Stage
client.transition_model_version_stage(
    name="llama-3.1-70b-instruct",
    version=3,
    stage="Production"
)
```

MLflow在2025年行业路线图中被定位为MLOps的基础要素。MLflow 3.0已将注册表扩展至生成式AI。

### 3.2 数据版本管理：DVC（Data Version Control）

模型的行为高度依赖于训练数据和超参数配置。如果数据版本不可追溯，模型的“好”与“坏”就成了无源之水。

**DVC的核心设计**：将实际数据存储于外部对象存储中，仅在Git中保留**指针文件（.dvc）** ，有效降低仓库膨胀速度。

```bash
# 添加数据集到DVC管理
dvc add data/train.jsonl

# 提交指针文件到Git
git add data/train.jsonl.dvc .dvc/config
git commit -m "v1.2 数据集更新"

# 切换数据集版本
git checkout v1.2
dvc checkout
```

**Git与DVC的分工**：
- **Git**：管理代码文件、配置文件及DVC元数据文件（.dvc）
- **DVC**：专注于大型数据集和模型文件的版本控制，处理数据缓存和远程同步

在LLM开发中，DVC与Git协同构建了完整的**数据版本控制体系**，解决了“代码与数据不同步”、“模型权重无法有效追踪”等核心痛点。2026年欧盟AI法案要求可审计的训练数据记录，使数据版本管理从“最佳实践”变成了**合规要求**。

---

## 四、可观测性三支柱：Metrics / Logs / Traces

可观测性不是把Metrics、Logs、Traces三件套堆在一起，而是让它们 **“互相说话”** 。传统监控给的是“表象”，可观测性揭示的是“因果”。

### 4.1 Metrics（指标）：Prometheus + Grafana

**Prometheus** 是核心指标采集和存储引擎，**Grafana** 是可视化与仪表盘平台。

**GPU层指标（DCGM Exporter）**

NVIDIA DCGM（Data Center GPU Manager）是官方硬件监控工具，提供纳秒级数据采集能力，覆盖**功率、温度、ECC错误等200+指标**。DCGM Exporter通过HTTP端点（`/metrics`）暴露GPU指标，供Prometheus抓取。

关键GPU指标：
- GPU利用率（`DCGM_FI_DEV_GPU_UTIL`）
- 显存使用量（`DCGM_FI_DEV_FB_USED`）
- 显存带宽利用率
- GPU温度与功耗
- PCIe传输速率

**推理服务层指标（vLLM / SGLang）**

vLLM通过`/metrics`端点暴露Prometheus格式的监控数据，包括：

| 指标 | 类型 | 含义 |
|------|------|------|
| `vllm:num_requests_running` | Gauge | 当前正在运行的请求数 |
| `vllm:num_requests_waiting` | Gauge | 等待队列中的请求数 |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT分布 |
| `vllm:time_per_output_token_seconds` | Histogram | TPOT分布 |
| `vllm:request_success_total` | Counter | 成功请求总数 |
| `vllm:spec_decode_acceptance_rate` | - | 推测解码接受率 |

SGLang同样通过Dynamo暴露Prometheus指标，并支持通过CUDA事件精确测量不同forward mode下的GPU执行时间。

**告警配置示例**（Prometheus AlertManager）：

```yaml
groups:
- name: llm_inference_alerts
  rules:
  - alert: HighTTFT
    expr: histogram_quantile(0.95, vllm:time_to_first_token_seconds) > 0.5
    for: 2m
    annotations:
      summary: "TTFT P95超过500ms"
  - alert: GPUUtilizationLow
    expr: avg(DCGM_FI_DEV_GPU_UTIL) < 30
    for: 10m
    annotations:
      summary: "GPU利用率持续低于30%，可能存在资源浪费"
  - alert: QueueBacklog
    expr: vllm:num_requests_waiting > 10
    for: 1m
    annotations:
      summary: "请求队列积压超过10个"
```

### 4.2 Logging（日志）：结构化日志 + ELK

日志记录了系统事件的**详细上下文**。对于推理服务，关键日志包括：

- **请求级日志**：每个请求的trace_id、模型名称、输入长度、输出长度、TTFT、TPOT
- **系统级日志**：GPU错误、OOM事件、调度器决策、KV Cache分配失败
- **模型级日志**：加载时间、量化配置、精度验证

**结构化日志**（JSON格式）便于ELK（Elasticsearch + Logstash + Kibana）栈进行检索和分析：

```json
{
  "timestamp": "2026-07-14T10:23:45.123Z",
  "level": "INFO",
  "trace_id": "a1b2c3d4e5f6",
  "service": "vllm-inference",
  "event": "request_completed",
  "model": "llama-3.1-70b",
  "prompt_tokens": 256,
  "generated_tokens": 512,
  "ttft_ms": 342,
  "tpot_ms": 45.6,
  "gpu_util_avg": 78.5
}
```

### 4.3 Tracing（追踪）：Jaeger + OpenTelemetry

分布式追踪记录请求在系统中的**完整路径**。对于推理服务，Tracing可以精确分解：

```
[API Gateway] → [负载均衡] → [推理引擎 Prefill] → [推理引擎 Decode Step 1] → ... → [推理引擎 Decode Step N] → [Response]
     ↑5ms            ↑2ms              ↑350ms                    ↑45ms                       ↑45ms
```

**Jaeger集成**的价值在于让推理服务不仅“性能拉满”，还能“看得见”每一次请求的完整旅程。

**核心机制**：
1. 为每个请求生成唯一 `TraceID`
2. 记录跨服务的 `Span` 信息（服务名、方法名、耗时、状态码等）
3. 通过 `TraceID` 串联整个链路上下游

**vLLM中的Tracing实现**：vLLM支持通过OpenTelemetry集成Jaeger，为每个请求创建完整的Trace，包含Prefill和每个Decode步骤的Span。最新版本还支持Prometheus **Exemplars** ——将请求ID附加到指标观测值上，实现**指标与特定请求的关联调试**。

```python
# vLLM启动时启用Tracing
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --enable-opentelemetry \
    --otlp-traces-endpoint http://jaeger-collector:4318/v1/traces
```

---

## 五、Golden Signals：推理服务的四大黄金指标

Google SRE团队提出的 **四大黄金信号（Four Golden Signals）** 是评估系统健康状态的关键框架：

### 5.1 延迟（Latency）

**服务一个请求需要多长时间**。

在推理服务中细分为：
- **TTFT（Time To First Token）** ：首Token延迟，直接影响用户体验
- **TPOT（Time Per Output Token）** ：每个输出Token的时间，决定流畅感
- **端到端延迟**：从请求到达到完整响应返回的总时间

**监控要点**：关注**分布**而非平均值——P50、P95、P99都要跟踪。

### 5.2 流量（Traffic）

**系统承受的请求量**。

关键指标：
- **QPS（Queries Per Second）** ：每秒请求数
- **输入Token吞吐量**：每秒处理的Prompt token数
- **输出Token吞吐量**：每秒生成的Token数
- **并发请求数**：当前正在处理的请求数量（`vllm:num_requests_running`）

### 5.3 错误（Errors）

**请求失败的比例**。

包括：
- HTTP 5xx错误率
- 超时请求比例
- 模型推理异常（OOM、CUDA错误）
- 调度器拒绝率（因显存不足而无法调度）

### 5.4 饱和度（Saturation）

**系统有多“满”** ——还剩多少余量。

推理服务的关键饱和度指标：
- **GPU利用率**：计算单元有多忙
- **显存使用率**：还剩多少显存给新请求
- **请求队列深度**：`vllm:num_requests_waiting`
- **KV Cache使用率**：PagedAttention的Block还有多少可用

**Golden Signals在推理服务中的落地**：

| Golden Signal | 推理服务对应指标 | 告警阈值（示例） |
|--------------|----------------|----------------|
| **延迟** | TTFT P95, TPOT P95 | TTFT P95 > 500ms |
| **流量** | QPS, Token/s | QPS突增200% |
| **错误** | 5xx率, 超时率 | 错误率 > 1% |
| **饱和度** | GPU Util, 显存使用率 | 显存使用率 > 90% |

---

## 六、总结：一张图看懂MLOps + 可观测性

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MLOps + 可观测性全景图                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    基础设施层（Kubernetes + GPU）                     │    │
│  │  Device Plugin（DaemonSet）→ Extended Resource → kube-scheduler     │    │
│  │  GPU共享：MIG（硬隔离） / Time Slicing（软隔离）                        │    │
│  │  弹性伸缩：HPA（QPS）+ VPA（内存）+ Cluster Autoscaler（节点）           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    资产管理层（实验 + 模型 + 数据）                      │   │
│  │  实验跟踪：MLflow（记录参数/指标/ artifacts）                            │   │
│  │  模型注册：MLflow Model Registry（版本 + Stage：Staging/Production）    │   │
│  │  数据版本：DVC（Git管理指针 + 外部存储存实际文件）                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 可观测性层（Metrics + Logs + Traces）                 │   │
│  │                                                                     │   │
│  │     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │     │   Metrics   │  │    Logs     │  │   Traces    │               │   │
│  │     │ Prometheus  │  │ 结构化JSON   │  │   Jaeger    │               │   │
│  │     │  + Grafana  │  │   + ELK     │  │OpenTelemetry│               │   │
│  │     └─────────────┘  └─────────────┘  └─────────────┘               │   │
│  │           ↓                 ↓                ↓                      │   │
│  │      DCGM（GPU硬件）    请求级事件记录    TraceID串联全链路              │   │
│  │      vLLM/SGLang指标    错误现场快照    Prefill/Decode分步耗时          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  Golden Signals（四大黄金指标）                        │   │
│  │    延迟（TTFT/TPOT） + 流量（QPS） + 错误（5xx率）+ 饱和度（显存/GPU）     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

**建议**：
1. **基础设施先行**：用NVIDIA GPU Operator统一管理驱动、Device Plugin和DCGM Exporter，别手工维护。
2. **资产可追溯是底线**：MLflow管模型版本和Stage，DVC管数据版本。2026年欧盟AI法案生效后，**数据版本可审计**将从建议变成强制要求。
3. **可观测性要“三件套联动”** ：Metrics告诉你“出事了”，Logs告诉你“发生了什么”，Traces告诉你“哪里出的事”。三者通过`trace_id`关联，才能快速定位根因。
