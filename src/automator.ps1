$File =".\output_powershell.txt"

Write-Output "Starting training at:" | Out-File $File
Get-Date | Out-File $File -Append

python .\main.py

Write-Output "" | Out-File $File -Append
Write-Output "Finished training at" | Out-File $File -Append
Get-Date | Out-File $File -Append

Write-Host "Task completed. Turning off..."
Stop-Computer -ComputerName localhost