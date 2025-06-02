Write-Host "🚀 Starting APOSSS System Tests..." -ForegroundColor Green
Write-Host ""

# Test if Flask app is running
Write-Host "🔍 Checking if Flask app is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -TimeoutSec 5
    Write-Host "✅ Flask app is running!" -ForegroundColor Green
} catch {
    Write-Host "❌ Flask app is not running. Please start it first with: python app.py" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🧪 Running User Management Tests..." -ForegroundColor Yellow
python test_user_system.py

Write-Host ""
Write-Host "🎯 Tests completed!" -ForegroundColor Green
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 