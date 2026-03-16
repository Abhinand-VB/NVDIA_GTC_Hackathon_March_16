# GPU Efficiency Advisor - Run script
# Uses Python 3.9 and installs deps if needed, then starts Streamlit.

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$Python = "c:\users\1abhi\appdata\local\programs\python\python39\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Python not found at: $Python"
    Write-Host "Trying: python from PATH"
    $Python = "python"
}

Set-Location $ProjectDir

Write-Host "Installing dependencies..."
& $Python -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed."
    exit 1
}

Write-Host "Starting Streamlit..."
& $Python -m streamlit run app.py --server.headless true
exit $LASTEXITCODE
