
$counters = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$results = @()
foreach ($c in $counters) {
    if ($c.CookedValue -gt 15MB) {
        $mb = [math]::Round($c.CookedValue / 1MB, 1)
        $inst = $c.InstanceName
        # Format usually: pid_1234_luid_..._phys_0
        if ($inst -match 'pid_(\d+)') {
            $pidNum = [int]$matches[1]
            $pName = (Get-Process -Id $pidNum -ErrorAction SilentlyContinue).ProcessName
            $results += [PSCustomObject]@{
                PID = $pidNum
                ProcessName = $pName
                VRAM_MB = $mb
                Instance = $inst
            }
        }
    }
}
$results | Sort-Object VRAM_MB -Descending | Select-Object -First 20 PID, ProcessName, VRAM_MB | Format-Table -AutoSize
