param(
  [string]$ClusterName = $(if ($env:CLUSTER_NAME) { $env:CLUSTER_NAME } else { "crowd-vision" })
)

$ErrorActionPreference = "Stop"
$KindContext = "kind-$ClusterName"

function Safe-Exec([scriptblock]$cmd) {
  try { & $cmd | Out-Host } catch { }
}

Write-Host "==> Delete kind cluster: $ClusterName"
Safe-Exec { kind delete cluster --name $ClusterName }

Write-Host "==> Remove leftover kind node containers (if any)"
$pattern = "^$ClusterName(-control-plane|-worker[0-9]*)?$"
docker ps -a --format '{{.Names}}' `
  | Select-String -Pattern $pattern `
  | ForEach-Object { docker rm -f $_.Matches[0].Value | Out-Host }

Write-Host "==> Clean kubeconfig entries"
Safe-Exec { kubectl config delete-context $KindContext }
Safe-Exec { kubectl config delete-cluster $KindContext }
Safe-Exec { kubectl config delete-user $KindContext }

Write-Host "==> Remaining kind clusters:"
Safe-Exec { kind get clusters }
Write-Host "==> Remaining kube contexts:"
Safe-Exec { kubectl config get-contexts }
