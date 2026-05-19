$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvpy = "C:\Users\elipp\OneDrive\Documents\Grocery Bagger\.venv\Scripts\python.exe"
& $venvpy ".\test_yolo_raft_pointcloud_pickplace_fast.py"
