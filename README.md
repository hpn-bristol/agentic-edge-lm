# Agentic AI for Edge Multi-Modal LM Orchestration

This repository contains the runtime implementation of a multi-agentic edge
multi-modal large model (LM) orchestration system. It includes the proposed
Multi-Agentic AI controller, the worker-side LM deployment controller, the
Lyapunov local deployment controller, LM inference services, and Kubernetes
deployment utilities.

## Repository Layout

```text
MA/ctdema/llm/              Global Planning Agent and Prompt Scheduling Agent
MA/ctdema/llm-route/        Minimal routing helpers used by the MA controller
MA/ctdema/runtime/EllmEnv.py Runtime environment used by the MA controller
MA/ctdema/exploration/       Initial macro-policy exploration examples
MA/onnode/sadrl/            Worker-side deployment API and local controllers
MA/onnode/inference/        LM inference services
MA/onnode/dockerfiles/      Dockerfiles for worker and LM service images
namespace/                   Kubernetes namespaces and node labels
RBAC/                        Kubernetes RBAC permissions
portforward/                 Host-network forwarding service
prometheus/                  Monitoring manifests
mongodb/                     MongoDB Helm chart used by the runtime telemetry
timeslice/                   GPU time-slicing configuration
monitorgpu/                  GPU monitoring helper
image_prepull/               Pre-pull and reset utility used before each run
scripts/                     Minimal run/reset utilities
deploy/rke/                  Sanitized RKE cluster template
```

## What Is Included

- Tier-1 Global Planning Agent.
- Tier-2 Prompt Scheduling Agent.
- Worker-side LM Deployment Control Agent.
- Lyapunov local deployment controller.
- FastAPI worker service for per-node LM activation and request handling.
- Four LM inference service entry points.
- Kubernetes manifests for namespace, RBAC, MongoDB, Prometheus, GPU
  time-slicing, and worker DaemonSet deployment.
- Runtime reset helper based on `image_prepull/preset.py`.

## Configuration

Copy the example environment file and fill in your local values:

```bash
cp .env.example .env
```

Important variables:

```bash
OPENAI_API_KEY=<your-openai-api-key>
LLM_POLICY_MODEL=<planner-model-name>
LOCAL_GPT_MODEL=<local-controller-model-name>
ELLM_STATE_DIR=./runtime_state
ELLM_POLICY_DIR=./runtime_state/llm_policy
IMAGE_PREFIX=<docker-user-or-registry>/elm
```

The controller writes runtime state under `runtime_state/` by default. This
directory is ignored by Git.

Initial macro-policy exploration examples are provided under
`MA/ctdema/exploration/`. They use placeholder node names and IP addresses;
edit them for your cluster before use. By default, the macro planner consumes
these example policies for the first macro epochs, then switches to live LLM
planning with the accumulated history. To use a different exploration directory:

```bash
export LLM_MACRO_EXPLORATION_DIR=<path-to-exploration-policies>
```

## Build Worker Images

Build all worker and LM images:

```bash
IMAGE_PREFIX=<docker-user-or-registry>/elm \
  ./MA/onnode/dockerfiles/build_and_push.sh
```

Push images after build:

```bash
DOCKER_USERNAME=<user> DOCKER_PASSWORD=<password> \
IMAGE_PREFIX=<docker-user-or-registry>/elm PUSH=1 \
  ./MA/onnode/dockerfiles/build_and_push.sh
```

Update image names in Kubernetes manifests if you use a different registry.

## Kubernetes Deployment

The deployment assumes that worker machines already have Docker, NVIDIA
drivers, and the NVIDIA container runtime installed where GPUs are available.
The RKE cluster template is provided in `deploy/rke/cluster.example.yml`.

Bring up the cluster and install the kubeconfig:

```bash
rke up --ignore-docker-version
mkdir -p ~/.kube
cp kube_config_cluster.yml ~/.kube/config
kubectl get nodes -o wide
```

Label worker nodes:

```bash
cd namespace
python3 labelnodes.py
cd ..
kubectl get nodes --show-labels
```

Create namespaces and image-pull secrets:

```bash
kubectl apply -f namespace/namespace.yml

kubectl create secret docker-registry llm-docker \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<docker-username> \
  --docker-password=<docker-password> \
  --docker-email=<docker-email> \
  -n worker

kubectl create secret docker-registry llm-docker \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<docker-username> \
  --docker-password=<docker-password> \
  --docker-email=<docker-email> \
  -n allworkers
```

Deploy runtime services and permissions:

```bash
helm install mongodb ./mongodb -n mongodb --create-namespace

kubectl apply -f RBAC/cluster-role.yml
kubectl apply -f RBAC/cluster-role-blinding.yml
kubectl apply -f portforward/forward-service.yml
kubectl apply -f prometheus/bundle.yaml
kubectl apply -f prometheus/node-exporter.yaml
kubectl apply -f prometheus/prometheus.yaml
kubectl apply -f timeslice/time-slicing-config.yml
```

If GPU time-slicing is managed by the NVIDIA GPU Operator, patch the operator
policy after applying the ConfigMap:

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n gpu-operator \
  --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config"}}}}'
```

Create the worker service account if it does not already exist:

```bash
kubectl create serviceaccount custom-sa -n allworkers
```

Deploy the worker-side controller:

```bash
kubectl apply -f MA/onnode/distributed-daemon.yml
kubectl rollout status daemonset/sadrl -n allworkers
kubectl get pods -n allworkers -o wide
```

Confirm permissions when needed:

```bash
kubectl auth can-i get nodes \
  --as=system:serviceaccount:allworkers:custom-sa

kubectl auth can-i get services \
  --namespace=mongodb \
  --as=system:serviceaccount:allworkers:custom-sa
```

## Reset Before a Run

Before launching a new MA run, clear controller-side state, MongoDB runtime
collections, worker-side FastAPI state, and LM image pre-pull state:

```bash
./scripts/reset_runtime_state.sh
```

This script calls:

```bash
scripts/reset_cluster_runtime.py
image_prepull/preset.py
```

The first script drops non-system MongoDB databases and calls `/cleanup` on each
worker service. The second script clears stale LM-worker deployments and ensures
the pre-pull DaemonSet is present.

The optional stale-image pruning helper uses SSH and is parameterized through
environment variables:

```bash
WORKER_HOSTS=<worker-ip-1>,<worker-ip-2> \
WORKER_SSH_USER=<ssh-user> \
WORKER_SSH_KEY=<path-to-private-key> \
IMAGE_PREFIX=<docker-user-or-registry>/elm \
  python3 image_prepull/prune_useless_images.py
```

## Run the MA Controller

Start one controller run:

```bash
./scripts/run_ma_once.sh
```

By default, the controller:

- starts the macro planner;
- spawns the slot-level router loop;
- sets the worker agents to Lyapunov/GPT-compatible runtime mode;
- writes telemetry to MongoDB using `LLM_ROUTING_DBNAME`.

To disable the Kubernetes monitor process for diagnostics:

```bash
LLM_SPAWN_MONITOR=0 ./scripts/run_ma_once.sh
```

## Debugging

Watch worker-side controller pods:

```bash
watch -n 2 'kubectl get pods -n allworkers -o wide'
```

Inspect one worker pod:

```bash
kubectl describe pod <sadrl-pod-name> -n allworkers
kubectl logs -f <sadrl-pod-name> -n allworkers
```

Check recent cluster events:

```bash
kubectl get events -A --sort-by=.lastTimestamp | tail -50
```

Check worker services and endpoints:

```bash
kubectl get svc,endpoints -n allworkers -o wide
kubectl get svc,endpoints -n mongodb -o wide
```

Check node labels and GPU resources:

```bash
kubectl get nodes --show-labels
kubectl describe node <worker-node-name> | grep -A8 -i "nvidia.com/gpu"
```

Check GPU Operator pods if GPU resources are missing:

```bash
kubectl get pods -n gpu-operator -o wide
kubectl logs -n gpu-operator <gpu-operator-pod-name>
```

Check RBAC permissions for the worker service account:

```bash
kubectl auth can-i get pods \
  --as=system:serviceaccount:allworkers:custom-sa \
  -n allworkers

kubectl auth can-i get services \
  --as=system:serviceaccount:allworkers:custom-sa \
  -n mongodb
```

Check controller-side logs:

```bash
tail -f runtime_state/*.log
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for
details.
