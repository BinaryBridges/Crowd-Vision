param(
  [string]$ClusterName    = $(if ($env:CLUSTER_NAME)   { $env:CLUSTER_NAME }   else { "dev" }),
  [string]$ImageTag       = $(if ($env:IMAGE_TAG)      { $env:IMAGE_TAG }      else { "your-app:dev" }),
  [string]$InstallMetrics = $(if ($env:INSTALL_METRICS){ $env:INSTALL_METRICS} else { "true" }),
  [string]$KustomizePath  = $(if ($env:KUSTOMIZE_PATH) { $env:KUSTOMIZE_PATH } else { "" })
)

$ErrorActionPreference = "Stop"

# Resolve repo root and default kustomize path
$ScriptDir = $PSScriptRoot
$RootDir   = (Resolve-Path "$ScriptDir\..").Path
if (-not $KustomizePath) { $KustomizePath = (Join-Path $RootDir "infra\k8s") }

function Assert-Cmd([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "$name not found in PATH"
  }
}

Write-Host "==> Preflight checks"
Assert-Cmd docker
Assert-Cmd kind
Assert-Cmd kubectl
docker ps | Out-Null

Write-Host "==> Creating/ensuring kind cluster: $ClusterName"
$clusters = (kind get clusters) -split "`n"
if ($clusters -notcontains $ClusterName) {
  kind create cluster --name $ClusterName --wait 120s | Out-Host
} else {
  # make sure node container is actually running
  $node = (docker ps --format '{{.Names}}' | Where-Object { $_ -eq "$ClusterName-control-plane" })
  if (-not $node) {
    Write-Host "Node container not running; recreating cluster..."
    kind delete cluster --name $ClusterName | Out-Host
    kind create cluster --name $ClusterName --wait 120s | Out-Host
  } else {
    Write-Host "Cluster exists and node is running."
  }
}

Write-Host "==> Set kubectl context: kind-$ClusterName"
kubectl config use-context "kind-$ClusterName" | Out-Host

Write-Host "==> Verify nodes"
kubectl get nodes -o wide | Out-Host

Write-Host "==> Build image: $ImageTag (context=$RootDir)"
docker build -t $ImageTag $RootDir | Out-Host

Write-Host "==> Load image into kind: $ClusterName"
kind load docker-image $ImageTag --name $ClusterName | Out-Host

if ($InstallMetrics -eq "true") {
  Write-Host "==> Install metrics-server (for HPA)"
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml | Out-Host
}

Write-Host "==> Apply kustomize: $KustomizePath"
kubectl apply -k "$KustomizePath" | Out-Host

Write-Host "==> Wait for app deployments (controller, worker)"
kubectl -n app rollout status deploy/controller --timeout=120s | Out-Host
kubectl -n app rollout status deploy/worker     --timeout=120s | Out-Host

Write-Host "==> Summary (namespace app)"
kubectl -n app get deploy,pod,svc,hpa,job,cronjob -o wide | Out-Host
