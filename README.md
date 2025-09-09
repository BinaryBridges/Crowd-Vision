<div align="center">

<pre>
################################################################################
#                                                                              #
#                                C R O W D   V I S I O N                       #
#                                                                              #
################################################################################
</pre>

</div>

### Steps to run

# 1) Start virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Ensure pip targets this interpreter
python -m pip install --upgrade pip setuptools wheel

# 3) Install DEV + base (dev file includes base)
python -m pip install -r .\requirements-dev.txt

3. Start project
`python main.py`

### Formatting and linting
ruff format .
ruff check --fix .







## How to run

```bash
# From the repo root (crowd-vision/)
# 0) prerequisites installed once: docker, kubectl, kind

# 1) Build the image locally
docker build -t your-app:dev .

# 2) Create a kind cluster
kind create cluster --name dev
kubectl config use-context kind-dev

# 3) Load the local image into the cluster (skips pushing to a registry)
kind load docker-image your-app:dev --name dev

# 4) (Optional) install metrics-server for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 5) Deploy everything
kubectl apply -k infra/k8s

# 6) Verify
kubectl -n app get all
kubectl -n app get deployments
kubectl -n app get jobs
kubectl -n app get cronjobs

# 7) Tail logs (they’ll be empty since functions are pass)
kubectl -n app logs deploy/controller
kubectl -n app logs deploy/worker -f

# 8) Trigger the one-off Job again (optional)
kubectl -n app delete job run-once && kubectl -n app apply -f infra/k8s/60-job-once.yaml

# 9) Clean up
# kind delete cluster --name dev

```

### Notes / knobs

Swap your-app:dev to a real registry image (e.g., ghcr.io/you/your-app:0.1.0) when you’re ready for cloud; remove the kind load step and ensure nodes can pull from your registry.

You can delete 40-service-optional.yaml until you expose a port.

HPA requires metrics-server; otherwise omit 50-hpa-workers.yaml.

The same image runs all roles; ROLE env var switches which blank function is called.

This is the minimal “everything in place” setup. If you want me to zip these into a gist-like blob you can download, say the word and I’ll paste a one-liner to recreate all files.