docker build --network host -t image.cestc.cn/ceai-internal/deep-research:v1.2.7 .

docker push image.cestc.cn/ceai-internal/deep-research:v1.2.7

kubectl --kubeconfig=/home/ceai/kubeconfig/config-dev apply -f /home/ceai/law_deep_research/deep-research-deploy.yaml

kubectl --kubeconfig=/home/ceai/kubeconfig/config-dev -n ceai logs -f deployment/deep-research --tail=50

# 在 deep_research_dev 目录下执行

python -m deep_research.run_server_streaming

python langchain/run_langgraph.py

情况已经查清，结论如下。

## 清空并重来

### 1. 本机先清干净

```bash
pkill -9 ngrok 2>/dev/null
pgrep ngrok || echo "本机已无 ngrok"
```

### 2. 在 ngrok 控制台断开远端 Agent

浏览器打开：

- **Agents**：https://dashboard.ngrok.com/agents  
  → 找到 **Online** 的 Agent → **Stop / Disconnect**
- **Endpoints**：https://dashboard.ngrok.com/endpoints  
  → 确认 `subchorionic-gilberto-trimodal.ngrok-free.dev` 已下线

### 3. 本机重新启动

```bash
ngrok http 8090
```

若旧隧道仍指向本机 8090，**可直接用现成地址**：

`https://subchorionic-gilberto-trimodal.ngrok-free.dev`
