# Where does a program's code use one of its message strings?  Reads a 64-bit Windows DLL as bytes (it runs nothing of
# it and writes nothing): finds the text, then every instruction that takes its address (lea reg, [rip+d]), and prints
# the bytes around each so the test that leads to the message can be read off.  Used on the game's server.dll to size
# the script compiler's limits ("Too many locals", "Too many consts"), which no script on disk could show.
#
#   pwsh tools\dll_string_xrefs.ps1 -Text "Too many locals" [-Dll <path>] [-Before 96] [-After 24]
#
# Many messages sit in a block of their own at the end of a routine, reached by a jump from the test.  -JumpTo lists
# the conditional jumps that land on such a block (give the address the block starts at, as the first call printed it):
#   pwsh tools\dll_string_xrefs.ps1 -JumpTo 0x50626
param(
    [string]$Text = '',
    [string]$Dll = 'D:\SteamLibrary\steamapps\common\Titanfall2\server.dll',
    [int]$Before = 96,
    [int]$After = 24,
    [long]$JumpTo = -1
)
$ErrorActionPreference = 'Stop'
$b = [System.IO.File]::ReadAllBytes($Dll)
$pe = [BitConverter]::ToInt32($b, 0x3C)
if ($b[$pe] -ne 0x50 -or $b[$pe + 1] -ne 0x45) { throw 'not a PE file' }
$nsec = [BitConverter]::ToUInt16($b, $pe + 6)
$opt = [BitConverter]::ToUInt16($b, $pe + 20)
$imageBase = [BitConverter]::ToUInt64($b, $pe + 24 + 24)
$secs = @()
for ($i = 0; $i -lt $nsec; $i++) {
    $o = $pe + 24 + $opt + 40 * $i
    $secs += [pscustomobject]@{
        Name  = [System.Text.Encoding]::ASCII.GetString($b, $o, 8).TrimEnd([char]0)
        VSize = [BitConverter]::ToUInt32($b, $o + 8)
        VAddr = [BitConverter]::ToUInt32($b, $o + 12)
        RSize = [BitConverter]::ToUInt32($b, $o + 16)
        RPtr  = [BitConverter]::ToUInt32($b, $o + 20)
        Chars = [BitConverter]::ToUInt32($b, $o + 36)
    }
}
function Rva([long]$off) { foreach ($s in $secs) { if ($off -ge $s.RPtr -and $off -lt $s.RPtr + $s.RSize) { return $off - $s.RPtr + $s.VAddr } }; return -1 }
"{0}: {1:N0} bytes, image base 0x{2:X}, sections: {3}" -f (Split-Path -Leaf $Dll), $b.Length, $imageBase, (($secs | ForEach-Object { "$($_.Name)@0x$('{0:X}' -f $_.VAddr)" }) -join ' ')

function Dump([long]$fileOff, $sec, [int]$len, [string]$label) {
    $start = [Math]::Max($sec.RPtr, $fileOff - $Before)
    for ($row = $start; $row -lt $fileOff + $len + $After; $row += 16) {
        $n = [Math]::Min(16, $fileOff + $len + $After - $row)
        $hex = ($b[$row..($row + $n - 1)] | ForEach-Object { '{0:X2}' -f $_ }) -join ' '
        $mark = if ($fileOff -ge $row -and $fileOff -lt $row + 16) { "  <- $label" } else { '' }
        "  {0:X8}  {1}{2}" -f ($row - $sec.RPtr + $sec.VAddr), $hex, $mark
    }
}

# the conditional (and plain) jumps that land on one address
if ($JumpTo -ge 0) {
    $found = 0
    foreach ($s in ($secs | Where-Object { $_.Chars -band 0x20000000 })) {
        $end = $s.RPtr + $s.RSize - 8
        for ($i = [int]$s.RPtr; $i -lt $end; $i++) {
            $op = $b[$i]
            $t = -1; $len = 0; $kind = ''
            if ($op -eq 0x0F -and ($b[$i + 1] -band 0xF0) -eq 0x80) { $t = ($i - $s.RPtr + $s.VAddr) + 6 + [BitConverter]::ToInt32($b, $i + 2); $len = 6; $kind = 'jcc 0F {0:X2} (rel32)' -f $b[$i + 1] }
            elseif (($op -band 0xF0) -eq 0x70) { $t = ($i - $s.RPtr + $s.VAddr) + 2 + [sbyte]$(if ($b[$i + 1] -gt 127) { $b[$i + 1] - 256 } else { $b[$i + 1] }); $len = 2; $kind = 'jcc {0:X2} (rel8)' -f $op }
            if ($t -eq $JumpTo) {
                $found++
                "--- jump {0}: {1} at rva 0x{2:X}" -f $found, $kind, ($i - $s.RPtr + $s.VAddr)
                Dump $i $s $len 'jump'
            }
        }
    }
    "jumps found: $found"
    return
}
if (-not $Text) { throw 'give -Text or -JumpTo' }

# the text, as a zero-ended ASCII string
$latin = [System.Text.Encoding]::GetEncoding(28591).GetString($b)
$needle = $Text + [char]0
$at = @()
$p = $latin.IndexOf($needle, [StringComparison]::Ordinal)
while ($p -ge 0) {
    # only where it starts a string (the byte before is a zero)
    if ($p -eq 0 -or $b[$p - 1] -eq 0) { $at += $p }
    $p = $latin.IndexOf($needle, $p + 1, [StringComparison]::Ordinal)
}
if ($at.Count -eq 0) { throw "the text is not in the file as a string of its own: $Text" }
$targets = @{}
foreach ($o in $at) { $r = Rva $o; $targets[[long]$r] = $o; "text at file offset 0x{0:X}, rva 0x{1:X}" -f $o, $r }

# every lea reg, [rip+disp32] in the code sections that lands on it
$found = 0
foreach ($s in ($secs | Where-Object { $_.Chars -band 0x20000000 })) {
    $end = $s.RPtr + $s.RSize - 8
    $i = [Array]::IndexOf($b, [byte]0x8D, [int]$s.RPtr)
    while ($i -ge 0 -and $i -lt $end) {
        if (($b[$i + 1] -band 0xC7) -eq 0x05) {
            $disp = [BitConverter]::ToInt32($b, $i + 2)
            $next = ($i - $s.RPtr + $s.VAddr) + 6
            $t = [long]$next + $disp
            if ($targets.ContainsKey($t)) {
                $found++
                $start = [Math]::Max($s.RPtr, $i - $Before)
                $rex = if (($b[$i - 1] -band 0xF0) -eq 0x40) { 1 } else { 0 }       # (the instruction starts at its REX prefix)
                "--- use {0}: lea at rva 0x{1:X} (file 0x{2:X}) in {3}" -f $found, ($i - $s.RPtr + $s.VAddr - $rex), ($i - $rex), $s.Name
                for ($row = $start; $row -lt $i + 6 + $After; $row += 16) {
                    $n = [Math]::Min(16, $i + 6 + $After - $row)
                    $hex = ($b[$row..($row + $n - 1)] | ForEach-Object { '{0:X2}' -f $_ }) -join ' '
                    $mark = if ($i -ge $row -and $i -lt $row + 16) { '  <- lea' } else { '' }
                    "  {0:X8}  {1}{2}" -f ($row - $s.RPtr + $s.VAddr), $hex, $mark
                }
            }
        }
        $i = [Array]::IndexOf($b, [byte]0x8D, $i + 1)
    }
}
"uses found: $found"
