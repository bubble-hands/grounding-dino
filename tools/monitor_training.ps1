$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$logFile = Join-Path $projectRoot "logs\training.log"
$checkInterval = 60

Write-Host "Training Monitor Started. Checking every $checkInterval seconds..."
Write-Host "Log file: $logFile"

while ($true) {
    Start-Sleep -Seconds $checkInterval
    
    if (Test-Path $logFile) {
        $content = Get-Content $logFile -Tail 5
        
        $lastEpoch = ""
        $lastTrainLoss = ""
        $lastValLoss = ""
        $earlyStop = $false
        
        foreach ($line in $content) {
            if ($line -match 'Epoch (\d+): Train Loss = ([\d.]+)') {
                $lastEpoch = $Matches[1]
                $lastTrainLoss = $Matches[2]
            }
            if ($line -match 'Epoch (\d+): Val Loss = ([\d.]+)') {
                $lastValLoss = $Matches[2]
            }
            if ($line -match 'Early stopping triggered') {
                $earlyStop = $true
            }
        }
        
        $time = Get-Date -Format "HH:mm:ss"
        $output = "[$time] "
        
        if ($lastEpoch) {
            $output += "Epoch $lastEpoch"
            if ($lastTrainLoss) {
                $output += " | Train Loss: $lastTrainLoss"
            }
            if ($lastValLoss) {
                $output += " | Val Loss: $lastValLoss"
            }
            if ($earlyStop) {
                $output += " | ** EARLY STOPPING **"
            }
        } else {
            $output += "Waiting for first epoch to complete..."
        }
        
        Write-Host $output
        
        if ($earlyStop) {
            Write-Host "`nTraining stopped due to early stopping!"
            break
        }
        
        if ($content -match 'Training completed') {
            Write-Host "`nTraining completed!"
            break
        }
    }
}
