# Packs what Ruche needs from this laptop (docs/RUCHE_RUNBOOK.md §2, §5, §6).
#
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_ruche.ps1 [-Code]
#
# * Always: the raw Nonastreda copy (labels, MAT file, tool/chip/work photos, MANIFEST.json)
#   from $env:RULFUSE_DATA_ROOT\raw\nonastreda (default D:\RULModel) as one tar archive
#   (about 1.2 GB; photos and the MAT file are already compressed, so no gzip).
# * With -Code: the committed state of the repository (git archive of HEAD), for when
#   Ruche cannot clone from GitHub. Uncommitted changes are NOT included: commit first.
#
# Use tar, not Explorer's "Compress" menu: zip files made on Windows unpack into wrongly
# named files on Linux (OmniFuser lesson).
param([switch]$Code)
$ErrorActionPreference = "Stop"

$root = if ($env:RULFUSE_DATA_ROOT) { $env:RULFUSE_DATA_ROOT } else { "D:\RULModel" }
$raw = Join-Path $root "raw"
$out = Join-Path $root "transfer"
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($name in @("labels.csv", "labels_reg.csv", "forces_xyz_raw.mat", "MANIFEST.json")) {
    if (-not (Test-Path (Join-Path $raw "nonastreda\$name"))) { throw "missing $raw\nonastreda\$name" }
}
foreach ($kind in @("tool", "chip", "work")) {
    $n = (Get-ChildItem (Join-Path $raw "nonastreda\$kind") -File).Count
    if ($n -ne 512) { throw "$kind has $n photos, expected 512" }
}

$data = Join-Path $out "nonastreda_raw.tar"
Write-Host "packing $raw\nonastreda -> $data"
tar -cf $data -C $raw nonastreda
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$hash = (Get-FileHash $data -Algorithm SHA256).Hash.ToLower()
# LF line ending: sha256sum -c on Linux would read a CR into the file name.
[IO.File]::WriteAllText((Join-Path $out "nonastreda_raw.tar.sha256"), "$hash  nonastreda_raw.tar`n")
Write-Host ("data archive: {0:N0} MB, sha256 {1}" -f ((Get-Item $data).Length / 1MB), $hash)

if ($Code) {
    $repo = Split-Path -Parent $PSScriptRoot
    $dirty = git -C $repo status --porcelain
    if ($dirty) { Write-Warning "uncommitted changes are not in the archive (commit them first)" }
    $archive = Join-Path $out "RUL-Model-code.tar.gz"
    git -C $repo archive --format=tar.gz --prefix=RUL-Model/ -o $archive HEAD
    Write-Host ("code archive: {0} ({1})" -f $archive, (git -C $repo log -1 --oneline))
}

Write-Host ""
Write-Host "Next (docs/RUCHE_RUNBOOK.md §6), with `$Ruche and `$RucheLogin set (§3):"
Write-Host "  scp `"$data`" `"$(Join-Path $out 'nonastreda_raw.tar.sha256')`" `"`${Ruche}:/workdir/`$RucheLogin/`""
