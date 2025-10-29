# Bak-Sneppen 3D Visualization - PowerShell Runner
# =================================================
# Quick launcher for rendering Bak-Sneppen animations on Windows

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('basic', 'enhanced', 'histogram', 'avalanche', 'all')]
    [string]$Scene = 'basic',
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('low', 'medium', 'high', '4k')]
    [string]$Quality = 'high',
    
    [Parameter(Mandatory=$false)]
    [switch]$NoPreview,
    
    [Parameter(Mandatory=$false)]
    [switch]$SaveLastFrame,
    
    [Parameter(Mandatory=$false)]
    [switch]$ExportGif
)

Write-Host "`n=== Bak-Sneppen 3D Visualization Renderer ===" -ForegroundColor Cyan
Write-Host "Scene: $Scene | Quality: $Quality`n" -ForegroundColor Yellow

# Map quality to manim flags
$qualityMap = @{
    'low' = '-ql'
    'medium' = '-qm'
    'high' = '-qh'
    '4k' = '-qk'
}
$qualityFlag = $qualityMap[$Quality]

# Add preview flag
$previewFlag = if (-not $NoPreview) { '-p' } else { '' }

# Add save last frame flag
$saveFrameFlag = if ($SaveLastFrame) { '-s' } else { '' }

# Add GIF export flag
$gifFlag = if ($ExportGif) { '--format=gif' } else { '' }

# Map scenes to class names and files
$sceneMap = @{
    'basic' = @{
        'file' = 'bak_sneppen_3d.py'
        'class' = 'BakSneppenEvolution3D'
    }
    'enhanced' = @{
        'file' = 'bak_sneppen_3d_enhanced.py'
        'class' = 'BakSneppenEnhanced'
    }
    'histogram' = @{
        'file' = 'bak_sneppen_3d.py'
        'class' = 'BakSneppenHistogram'
    }
    'avalanche' = @{
        'file' = 'bak_sneppen_3d.py'
        'class' = 'BakSneppenAvalanche'
    }
}

function Render-Scene {
    param($File, $Class)
    
    Write-Host "`nRendering $Class from $File..." -ForegroundColor Green
    
    $command = "manim $qualityFlag $previewFlag $saveFrameFlag $gifFlag $File $Class"
    Write-Host "Command: $command`n" -ForegroundColor DarkGray
    
    # Execute manim command
    Invoke-Expression $command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Successfully rendered $Class`n" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Failed to render $Class (exit code: $LASTEXITCODE)`n" -ForegroundColor Red
    }
}

# Check if manim is installed
try {
    $manimVersion = manim --version 2>&1
    Write-Host "Found: $manimVersion`n" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Manim is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Install with: pip install manim`n" -ForegroundColor Yellow
    exit 1
}

# Render requested scenes
if ($Scene -eq 'all') {
    Write-Host "Rendering all scenes...`n" -ForegroundColor Cyan
    
    foreach ($key in $sceneMap.Keys) {
        $sceneInfo = $sceneMap[$key]
        Render-Scene -File $sceneInfo['file'] -Class $sceneInfo['class']
    }
} else {
    if ($sceneMap.ContainsKey($Scene)) {
        $sceneInfo = $sceneMap[$Scene]
        Render-Scene -File $sceneInfo['file'] -Class $sceneInfo['class']
    } else {
        Write-Host "ERROR: Unknown scene '$Scene'" -ForegroundColor Red
        Write-Host "Available scenes: basic, enhanced, histogram, avalanche, all`n" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "`n=== Rendering Complete ===" -ForegroundColor Cyan
Write-Host "Check the 'media' folder for output files`n" -ForegroundColor Yellow

# Usage examples
Write-Host "USAGE EXAMPLES:" -ForegroundColor Cyan
Write-Host "  .\run_bak_sneppen.ps1                           # Render basic scene in high quality"
Write-Host "  .\run_bak_sneppen.ps1 -Scene enhanced -Quality medium"
Write-Host "  .\run_bak_sneppen.ps1 -Scene all -Quality low"
Write-Host "  .\run_bak_sneppen.ps1 -Scene basic -ExportGif"
Write-Host "  .\run_bak_sneppen.ps1 -Scene histogram -NoPreview -SaveLastFrame`n"

