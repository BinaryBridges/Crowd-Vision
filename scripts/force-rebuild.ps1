param(
  [string]$ClusterName = $(if ($env:CLUSTER_NAME) { $env:CLUSTER_NAME } else { "crowd-vision" }),
  [string]$ImageTag    = $(if ($env:IMAGE_TAG)     { $env:IMAGE_TAG }     else { "your-app:dev" })
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RootDir   = (Resolve-Path "$ScriptDir\..").Path

Write-Host "==> FORCE REBUILD - Clearing all caches" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clear Python cache files
Write-Host "Step 1: Clearing Python __pycache__ directories..." -ForegroundColor Yellow
Get-ChildItem -Path "$RootDir\app" -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem -Path "$RootDir\app" -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "  ✓ Python cache cleared" -ForegroundColor Green

# Step 2: Remove old Docker image
Write-Host ""
Write-Host "Step 2: Removing old Docker image..." -ForegroundColor Yellow
docker rmi $ImageTag -f 2>$null | Out-Null
Write-Host "  ✓ Old image removed" -ForegroundColor Green

# Step 3: Build fresh image with --no-cache
Write-Host ""
Write-Host "Step 3: Building fresh Docker image (no cache)..." -ForegroundColor Yellow
Write-Host "  Image: $ImageTag" -ForegroundColor Gray
Write-Host "  Context: $RootDir" -ForegroundColor Gray
docker build --no-cache -t $ImageTag $RootDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "  X Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  + Image built successfully" -ForegroundColor Green

# Step 4: Verify the config in the image
Write-Host ""
Write-Host "Step 4: Verifying config in Docker image..." -ForegroundColor Yellow
$configCheck = docker run --rm $ImageTag python -c 'from app import config; print(config.CONVEX_BASE_URL)'
Write-Host "  CONVEX_BASE_URL in image: $configCheck" -ForegroundColor Gray
if ($configCheck -like "*3210*") {
    Write-Host "  + Config verified - using port 3210" -ForegroundColor Green
} else {
    Write-Host "  X WARNING: Config shows port $configCheck instead of 3210!" -ForegroundColor Red
}

# Step 5: Load into kind cluster
Write-Host ""
Write-Host "Step 5: Loading image into kind cluster..." -ForegroundColor Yellow
kind load docker-image $ImageTag --name $ClusterName
if ($LASTEXITCODE -ne 0) {
    Write-Host "  X Failed to load image into kind!" -ForegroundColor Red
    exit 1
}
Write-Host "  + Image loaded into cluster" -ForegroundColor Green

# Step 6: Delete existing pods to force recreation
Write-Host ""
Write-Host "Step 6: Deleting existing pods to force recreation..." -ForegroundColor Yellow
kubectl -n app delete pods --all 2>$null | Out-Null
Write-Host "  + Pods deleted" -ForegroundColor Green

# Step 7: Wait for new pods
Write-Host ""
Write-Host "Step 7: Waiting for new pods to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
kubectl -n app wait --for=condition=ready pod --all --timeout=120s 2>$null | Out-Null
Write-Host "  + Pods ready" -ForegroundColor Green

# Step 8: Show status
Write-Host ""
Write-Host "==> REBUILD COMPLETE" -ForegroundColor Cyan
Write-Host ""
Write-Host "Current pods:" -ForegroundColor Yellow
kubectl -n app get pods

Write-Host ""
Write-Host "To verify the Convex URL in a running pod, run:" -ForegroundColor Yellow
Write-Host '  kubectl -n app exec -it deploy/controller -- python -c "from app import config; print(config.CONVEX_BASE_URL)"' -ForegroundColor Gray

