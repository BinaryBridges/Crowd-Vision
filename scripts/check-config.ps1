param(
  [string]$ImageTag = $(if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "your-app:dev" })
)

Write-Host "==> Checking Convex Configuration" -ForegroundColor Cyan
Write-Host ""

# Check 1: Local file
Write-Host "1. Checking local app/config.py file:" -ForegroundColor Yellow
$localConfig = Get-Content "app/config.py" | Select-String "CONVEX_BASE_URL"
Write-Host "   $localConfig" -ForegroundColor Gray

# Check 2: Docker image (if it exists)
Write-Host ""
Write-Host "2. Checking config in Docker image '$ImageTag':" -ForegroundColor Yellow
try {
    $imageConfig = docker run --rm $ImageTag python -c 'from app import config; print(f"CONVEX_BASE_URL={config.CONVEX_BASE_URL}")' 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   $imageConfig" -ForegroundColor Gray
        if ($imageConfig -like "*3210*") {
            Write-Host "   + Image has correct port (3210)" -ForegroundColor Green
        } else {
            Write-Host "   X Image has WRONG port!" -ForegroundColor Red
            Write-Host "   -> You need to rebuild the image" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   X Image not found or error running" -ForegroundColor Red
    }
} catch {
    Write-Host "   X Could not check image" -ForegroundColor Red
}

# Check 3: Kubernetes ConfigMap
Write-Host ""
Write-Host "3. Checking Kubernetes ConfigMap:" -ForegroundColor Yellow
try {
    $k8sConfig = kubectl -n app get configmap app-config -o jsonpath='{.data.CONVEX_BASE_URL}' 2>$null
    if ($LASTEXITCODE -eq 0 -and $k8sConfig) {
        Write-Host "   CONVEX_BASE_URL=$k8sConfig" -ForegroundColor Gray
        if ($k8sConfig -like "*3210*") {
            Write-Host "   + ConfigMap has correct port (3210)" -ForegroundColor Green
        } else {
            Write-Host "   X ConfigMap has WRONG port!" -ForegroundColor Red
        }
    } else {
        Write-Host "   X ConfigMap not found or cluster not running" -ForegroundColor Red
    }
} catch {
    Write-Host "   X Could not check ConfigMap" -ForegroundColor Red
}

# Check 4: Running pod (if any)
Write-Host ""
Write-Host "4. Checking running controller pod:" -ForegroundColor Yellow
try {
    $podConfig = kubectl -n app exec deploy/controller -- python -c 'from app import config; print(config.CONVEX_BASE_URL)' 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   CONVEX_BASE_URL=$podConfig" -ForegroundColor Gray
        if ($podConfig -like "*3210*") {
            Write-Host "   + Running pod has correct port (3210)" -ForegroundColor Green
        } else {
            Write-Host "   X Running pod has WRONG port!" -ForegroundColor Red
            Write-Host "   -> The pod is using an old image" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   X No running controller pod found" -ForegroundColor Red
    }
} catch {
    Write-Host "   X Could not check running pod" -ForegroundColor Red
}

Write-Host ""
Write-Host "==> Summary" -ForegroundColor Cyan
Write-Host "If any checks show the wrong port, run:" -ForegroundColor Yellow
Write-Host "  .\scripts\force-rebuild.ps1" -ForegroundColor White

