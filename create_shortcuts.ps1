Param(
    [switch] $Force
)

# Create Desktop shortcuts for run_live.bat and run_pipeline.bat
$project = Split-Path -Parent $MyInvocation.MyCommand.Definition
$desktop = [Environment]::GetFolderPath("Desktop")

$shortcuts = @{
    "Run Live" = "run_live.bat"
    "Run Pipeline" = "run_pipeline.bat"
}

$w = New-Object -ComObject WScript.Shell

foreach ($name in $shortcuts.Keys) {
    $targetRel = $shortcuts[$name]
    $target = Join-Path $project $targetRel
    $linkPath = Join-Path $desktop ("$name.lnk")

    if (-not (Test-Path $target)) {
        Write-Output "Warning: target not found: $target"
        continue
    }

    if (Test-Path $linkPath) {
        if ($Force) {
            Remove-Item $linkPath -Force
        }
        else {
            Write-Output "Shortcut already exists: $linkPath (use -Force to overwrite)"
            continue
        }
    }

    $sc = $w.CreateShortcut($linkPath)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = $project
    $sc.IconLocation = $target
    $sc.Save()

    Write-Output "Created shortcut: $linkPath -> $target"
}

Write-Output "Done."
