# Contact sheets of a game clip without Python or ffmpeg (2026-11-19: starting python.exe blue-screens this PC).
# A tiny web server on localhost only: it serves one video file (with byte ranges, so the browser can seek) and a
# page that seeks through it, draws the frames into sheets and posts them back here to be saved as JPEGs.
# The browser that decodes the video is the desktop app's built-in one, already running.
#   pwsh -File tools/clipserver.ps1 -Clip <video.mp4> -Out <folder> [-Port 8765]
param(
    [Parameter(Mandatory = $true)] [string] $Clip,
    [Parameter(Mandatory = $true)] [string] $Out,
    [int] $Port = 8765
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Clip)) { throw "no such clip: $Clip" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$page = Join-Path $PSScriptRoot 'clipsheets.html'
$len = (Get-Item -LiteralPath $Clip).Length
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("http://localhost:$Port/")
$listener.Start()
Write-Host "clipserver: http://localhost:$Port/  clip $Clip ($len bytes)  out $Out"
$chunk = 16MB
while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    $req = $ctx.Request
    $res = $ctx.Response
    try {
        $path = $req.Url.AbsolutePath
        if ($req.HttpMethod -eq 'GET' -and ($path -eq '/' -or $path -eq '/index.html')) {
            $bytes = [IO.File]::ReadAllBytes($page)
            $res.ContentType = 'text/html; charset=utf-8'
            $res.ContentLength64 = $bytes.Length
            $res.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($req.HttpMethod -eq 'GET' -and $path -eq '/clip.mp4') {
            $from = 0L; $to = $len - 1
            $range = $req.Headers['Range']
            if ($range -match '^bytes=(\d*)-(\d*)$') {
                if ($Matches[1] -ne '') { $from = [long]$Matches[1] }
                if ($Matches[2] -ne '') { $to = [long]$Matches[2] }
            }
            if ($to -ge $len) { $to = $len - 1 }
            if ($to - $from + 1 -gt $chunk) { $to = $from + $chunk - 1 }   # at most 16 MB a reply: the browser asks again
            $res.StatusCode = 206
            $res.ContentType = 'video/mp4'
            $res.AddHeader('Accept-Ranges', 'bytes')
            $res.AddHeader('Content-Range', "bytes $from-$to/$len")
            $res.ContentLength64 = $to - $from + 1
            $fs = [IO.File]::OpenRead($Clip)
            try {
                $fs.Seek($from, 'Begin') | Out-Null
                $buf = New-Object byte[] 1MB
                $left = $to - $from + 1
                while ($left -gt 0) {
                    $n = $fs.Read($buf, 0, [int][Math]::Min($buf.Length, $left))
                    if ($n -le 0) { break }
                    $res.OutputStream.Write($buf, 0, $n)
                    $left -= $n
                }
            } finally { $fs.Close() }
        }
        elseif ($req.HttpMethod -eq 'POST' -and $path -eq '/save') {
            $name = $req.QueryString['name']
            if ($name -notmatch '^[A-Za-z0-9_.-]{1,64}$') { throw "bad name" }
            $ms = [IO.MemoryStream]::new()
            $req.InputStream.CopyTo($ms)
            [IO.File]::WriteAllBytes((Join-Path $Out $name), $ms.ToArray())
            Write-Host "saved $name ($($ms.Length) bytes)"
            $res.StatusCode = 204
        }
        elseif ($req.HttpMethod -eq 'POST' -and $path -eq '/log') {
            $sr = [IO.StreamReader]::new($req.InputStream)
            $text = $sr.ReadToEnd()
            Add-Content -LiteralPath (Join-Path $Out 'sheets_log.txt') -Value $text
            $res.StatusCode = 204
        }
        else {
            $res.StatusCode = 404
        }
    }
    catch {
        try { $res.StatusCode = 500 } catch {}
    }
    finally {
        try { $res.Close() } catch {}
    }
}
