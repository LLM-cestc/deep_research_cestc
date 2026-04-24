
docker build --network host -t image.cestc.cn/ceai-internal/deep-research:v1.2.7 .

docker push image.cestc.cn/ceai-internal/deep-research:v1.2.7

kubectl --kubeconfig=/home/ceai/kubeconfig/config-dev apply -f /home/ceai/law_deep_research/deep-research-deploy.yaml

kubectl --kubeconfig=/home/ceai/kubeconfig/config-dev -n ceai logs -f deployment/deep-research --tail=50

http://10.252.216.16:3459/

python -m deep_research.run_server_streaming
 end
