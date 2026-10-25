
param(
    [string]$DatasetRoot = 'D:\YOLO\data\processed',

    [string]$QuarantineRoot = 'D:\YOLO\data\quarantine',

    [ValidateSet('train', 'val', 'test')]
    [string]$Split = 'train',

    [string]$LogFile = 'D:\YOLO\logs\removal_log.csv',

    [int]$ExpectedCount = -1,

    [switch]$Apply
)

$imageDir = Join-Path $DatasetRoot "images\$Split"
$labelDir = Join-Path $DatasetRoot "labels\$Split"
$quarantineDir = Join-Path $QuarantineRoot "orphan_labels_$Split"

if (-not (Test-Path $imageDir) -or
    -not (Test-Path $labelDir)) {
    throw "Image or label directory does not exist for $Split"
}

$images = @(
    Get-ChildItem $imageDir -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in
        @('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    }
)

$imageStems = @(
    $images | ForEach-Object { $_.BaseName }
)

$orphans = @(
    Get-ChildItem $labelDir -File -Filter '*.txt' |
    Where-Object { $_.BaseName -notin $imageStems }
)

Write-Host "Dataset: $DatasetRoot"
Write-Host "Split: $Split"
Write-Host "Orphan labels found: $($orphans.Count)"

if ($ExpectedCount -ge 0 -and
    $orphans.Count -ne $ExpectedCount) {
    throw "Expected $ExpectedCount orphan labels, found $($orphans.Count). Nothing moved."
}

if ($orphans.Count -eq 0) {
    Write-Host 'No orphan labels to move.'
    exit 0
}

$orphans | Select-Object Name, FullName

if (-not $Apply) {
    Write-Host 'DRY RUN: no files moved. Add -Apply to execute.'
    exit 0
}

if (-not (Test-Path $quarantineDir)) {
    New-Item -ItemType Directory -Path $quarantineDir -Force |
        Out-Null
}

$timestamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'
$records = @()

foreach ($file in $orphans) {
    $destination = Join-Path $quarantineDir $file.Name

    if (Test-Path $destination) {
        throw "Destination already exists: $destination. Nothing further will be attempted."
    }

    Move-Item -LiteralPath $file.FullName `
        -Destination $destination `
        -ErrorAction Stop

    $records += [pscustomobject]@{
        timestamp = $timestamp
        split     = $Split
        file      = $file.Name
        source    = $file.FullName
        quarantine = $destination
        reason    = 'No image with matching filename stem'
    }
}

if ($records.Count -gt 0) {
    if (Test-Path $LogFile) {
        $records | Export-Csv -LiteralPath $LogFile `
            -Append -NoTypeInformation
    }
    else {
        $records | Export-Csv -LiteralPath $LogFile `
            -NoTypeInformation
    }
}

Write-Host "Moved $($records.Count) orphan labels."
Write-Host "Log: $LogFile"
