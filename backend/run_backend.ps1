$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

if (-not $env:OLLAMA_HOST) {
    $env:OLLAMA_HOST = "http://127.0.0.1:11434"
}
if (-not $env:GEMMA_FAST_MODEL) {
    $env:GEMMA_FAST_MODEL = "gemma4:e2b"
}
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $scriptRoot ".hf-cache"
}
$env:ANONYMIZED_TELEMETRY = "False"
if (-not $env:FAST_BACKEND_PORT) {
    $env:FAST_BACKEND_PORT = "8011"
}
if (-not $env:FAST_USE_MINILM) {
    $env:FAST_USE_MINILM = "0"
}
if (-not $env:FAST_SKIP_GEMMA_ON_UPLOAD) {
    $env:FAST_SKIP_GEMMA_ON_UPLOAD = "1"
}

.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port $env:FAST_BACKEND_PORT
