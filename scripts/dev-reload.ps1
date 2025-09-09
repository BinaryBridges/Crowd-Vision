param(
  [string]$ClusterName = $(if ($env:CLUSTER_NAME) { $env:CLUSTER_NAME } else { "dev" }),
  [string]$ImageTag    = $(if ($env:IMAGE_TAG)     { $env:IMAGE_TAG }     else { "your-app:dev" })
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RootDir   = (Resolve-Path "$ScriptDir\..").Path

function Assert-Cmd([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "$name not found in PATH"
  }
}

Assert-Cmd docker
Assert-Cmd kind
Assert-Cmd kubectl

Write-Host "==> Rebuild image: $ImageTag (context=$RootDir)"
docker build -t $ImageTag $RootDir | Out-Host

Write-Host "==> Load image into kind: $ClusterName"
kind load docker-image $ImageTag --name $ClusterName | Out-Host

Write-Host "==> Rollout restart deployments"
kubectl -n app rollout restart deploy/controller | Out-Host
kubectl -n app rollout restart deploy/worker     | Out-Host

Write-Host "==> Wait for rollouts"
kubectl -n app rollout status deploy/controller --timeout=120s | Out-Host
kubectl -n app rollout status deploy/worker     --timeout=120s | Out-Host

Write-Host "==> Done."
