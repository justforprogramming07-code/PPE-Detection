
param(
    [string]$DatasetRoot = 'D:\YOLO\data\processed'
)

$fail = $false
$imageExtensions = @('.jpg', '.jpeg', '.png', '.bmp', '.webp')

foreach ($split in @('train', 'val', 'test')) {
    $imageDir = Join-Path $DatasetRoot "images\$split"
    $labelDir = Join-Path $DatasetRoot "labels\$split"

    if (-not (Test-Path $imageDir) -or
        -not (Test-Path $labelDir)) {
        Write-Error "Missing image or label directory for $split"
        $fail = $true
        continue
    }

    $images = @(
        Get-ChildItem $imageDir -File |
        Where-Object {
            $_.Extension.ToLowerInvariant() -in $imageExtensions
        }
    )

    $labels = @(
        Get-ChildItem $labelDir -File -Filter '*.txt'
    )

    $imageStems = @(
        $images | ForEach-Object { $_.BaseName }
    )

    $labelStems = @(
        $labels | ForEach-Object { $_.BaseName }
    )

    $orphans = @(
        $labels | Where-Object {
            $_.BaseName -notin $imageStems
        }
    )

    $unlabeledImages = @(
        $images | Where-Object {
            $_.BaseName -notin $labelStems
        }
    )

    Write-Host "$split : images=$($images.Count) labels=$($labels.Count) orphan_labels=$($orphans.Count) images_without_labels=$($unlabeledImages.Count)"

    if ($orphans.Count -gt 0) {
        $fail = $true
    }
}

if ($fail) {
    Write-Host 'VALIDATION FAILED'
    exit 1
}

Write-Host 'VALIDATION PASSED'
exit 0
