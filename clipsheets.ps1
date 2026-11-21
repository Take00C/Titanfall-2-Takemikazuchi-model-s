# Contact sheets of a game clip without Python or ffmpeg (2026-11-19: starting python.exe blue-screens this PC).
# Decodes in this PowerShell process with Windows Media Foundation (the decoder Explorer uses for video thumbnails;
# HEVC needs the HEVC Video Extensions), on the CPU only: hardware decoders and DXVA are switched off.
# One pass through the clip; frames every -Step seconds go into detail sheets (d_NNN.jpg, -Per frames each: frame k of
# sheet N at t = (N-1)*Per*Step + k*Step) and every -OverEvery-th of them into overview sheets (o_NNN.jpg).
#   pwsh -NoProfile -File tools/clipsheets.ps1 -Clip <video.mp4> -Out <folder> [-Step 0.25] [-Per 12] [-OverEvery 4]
param(
    [Parameter(Mandatory = $true)] [string] $Clip,
    [Parameter(Mandatory = $true)] [string] $Out,
    [double] $Step = 0.25,
    [int] $Per = 12,
    [int] $OverEvery = 4,
    [int] $CellWidth = 640,
    [int] $Cols = 4,
    [double] $MaxSeconds = 0
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Clip)) { throw "no such clip: $Clip" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace TsfClip
{
    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public long hVal; }

    // vtable order as in mfobjects.h / mfreadwrite.h; methods never called keep their slot with an empty signature
    [ComImport, Guid("44ae0fa8-ea31-4109-8d2e-4cae4997c555"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMFMediaType
    {
        void GetItem(); void GetItemType(); void CompareItem(); void Compare();
        void GetUINT32([In] ref Guid key, out uint value);
        void GetUINT64([In] ref Guid key, out ulong value);
        void GetDouble(); void GetGUID(); void GetStringLength(); void GetString(); void GetAllocatedString();
        void GetBlobSize(); void GetBlob(); void GetAllocatedBlob(); void GetUnknown(); void SetItem(); void DeleteItem();
        void DeleteAllItems();
        void SetUINT32([In] ref Guid key, uint value);
        void SetUINT64([In] ref Guid key, ulong value);
        void SetDouble();
        void SetGUID([In] ref Guid key, [In] ref Guid value);
        void SetString(); void SetBlob(); void SetUnknown(); void LockStore(); void UnlockStore(); void GetCount();
        void GetItemByIndex(); void CopyAllItems();
        void GetMajorType(); void IsCompressedFormat(); void IsEqual(); void GetRepresentation(); void FreeRepresentation();
    }

    [ComImport, Guid("2cd2d921-c447-44a7-a13c-4adabfc247e3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMFAttributes
    {
        void GetItem(); void GetItemType(); void CompareItem(); void Compare();
        void GetUINT32([In] ref Guid key, out uint value);
        void GetUINT64([In] ref Guid key, out ulong value);
        void GetDouble(); void GetGUID(); void GetStringLength(); void GetString(); void GetAllocatedString();
        void GetBlobSize(); void GetBlob(); void GetAllocatedBlob(); void GetUnknown(); void SetItem(); void DeleteItem();
        void DeleteAllItems();
        void SetUINT32([In] ref Guid key, uint value);
        void SetUINT64([In] ref Guid key, ulong value);
        void SetDouble();
        void SetGUID([In] ref Guid key, [In] ref Guid value);
        void SetString(); void SetBlob(); void SetUnknown(); void LockStore(); void UnlockStore(); void GetCount();
        void GetItemByIndex(); void CopyAllItems();
    }

    [ComImport, Guid("045fa593-8799-42b8-bc8d-8968c6453507"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMFMediaBuffer
    {
        void Lock(out IntPtr buffer, out uint maxLength, out uint currentLength);
        void Unlock();
        void GetCurrentLength(out uint length);
        void SetCurrentLength(uint length);
        void GetMaxLength(out uint length);
    }

    [ComImport, Guid("c40a00f2-b93a-4d80-ae8c-5a1c634f58e4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMFSample
    {
        void GetItem(); void GetItemType(); void CompareItem(); void Compare(); void GetUINT32(); void GetUINT64();
        void GetDouble(); void GetGUID(); void GetStringLength(); void GetString(); void GetAllocatedString();
        void GetBlobSize(); void GetBlob(); void GetAllocatedBlob(); void GetUnknown(); void SetItem(); void DeleteItem();
        void DeleteAllItems(); void SetUINT32(); void SetUINT64(); void SetDouble(); void SetGUID(); void SetString();
        void SetBlob(); void SetUnknown(); void LockStore(); void UnlockStore(); void GetCount(); void GetItemByIndex();
        void CopyAllItems();
        void GetSampleFlags(); void SetSampleFlags(); void GetSampleTime(); void SetSampleTime(); void GetSampleDuration();
        void SetSampleDuration(); void GetBufferCount(); void GetBufferByIndex();
        void ConvertToContiguousBuffer(out IMFMediaBuffer buffer);
        void AddBuffer(); void RemoveBufferByIndex(); void RemoveAllBuffers(); void GetTotalLength(); void CopyToBuffer();
    }

    [ComImport, Guid("70ae66f2-c809-4e4f-8915-bdcb406b7993"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMFSourceReader
    {
        void GetStreamSelection();
        void SetStreamSelection(uint streamIndex, [MarshalAs(UnmanagedType.Bool)] bool selected);
        void GetNativeMediaType();
        void GetCurrentMediaType(uint streamIndex, out IMFMediaType mediaType);
        void SetCurrentMediaType(uint streamIndex, IntPtr reserved, IMFMediaType mediaType);
        void SetCurrentPosition([In] ref Guid timeFormat, [In] ref PropVariant position);
        void ReadSample(uint streamIndex, uint controlFlags, out uint actualStreamIndex, out uint streamFlags,
                        out long timestamp, out IMFSample sample);
        void Flush(); void GetServiceForStream(); void GetPresentationAttribute();
    }

    public sealed class Reader : IDisposable
    {
        [DllImport("mfplat.dll", PreserveSig = false)] static extern void MFStartup(uint version, uint flags);
        [DllImport("mfplat.dll", PreserveSig = false)] static extern void MFShutdown();
        [DllImport("mfplat.dll", PreserveSig = false)] static extern void MFCreateAttributes(out IMFAttributes attributes, uint size);
        [DllImport("mfplat.dll", PreserveSig = false)] static extern void MFCreateMediaType(out IMFMediaType mediaType);
        [DllImport("mfreadwrite.dll", PreserveSig = false)]
        static extern void MFCreateSourceReaderFromURL([MarshalAs(UnmanagedType.LPWStr)] string url, IMFAttributes attributes,
                                                        out IMFSourceReader reader);

        const uint FIRST_VIDEO = 0xFFFFFFFC, ALL_STREAMS = 0xFFFFFFFE;
        static Guid VIDEO_PROCESSING = new Guid("fb394f3d-ccf1-42ee-bbb3-f9b845d5681d"); // MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING
        static Guid DISABLE_DXVA = new Guid("aa456cfd-3943-4a1e-a77d-1838c0ea2e35");     // MF_SOURCE_READER_DISABLE_DXVA
        static Guid HW_TRANSFORMS = new Guid("a634a91c-822b-41b9-a494-4de4643612b0");    // MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS
        static Guid MT_MAJOR = new Guid("48eba18e-f8c9-4687-bf11-0a74c9f96a8f");
        static Guid MT_SUBTYPE = new Guid("f7e34c9a-42e8-4714-b74b-cb29d72c35e5");
        static Guid MT_FRAME_SIZE = new Guid("1652c33d-d6b2-4012-b834-72030849a37d");
        static Guid MT_DEFAULT_STRIDE = new Guid("644b4e48-1e02-4516-b0eb-c01ca9d49ac6");
        static Guid VIDEO = new Guid("73646976-0000-0010-8000-00aa00389b71");
        static Guid RGB32 = new Guid("00000016-0000-0010-8000-00aa00389b71");

        IMFSourceReader reader;
        IMFSample held;
        byte[] frame;
        public int W, H, Stride;
        public long Timestamp;

        public Reader(string path)
        {
            MFStartup(0x20070, 0);
            IMFAttributes a;
            MFCreateAttributes(out a, 4);
            a.SetUINT32(ref VIDEO_PROCESSING, 1);
            a.SetUINT32(ref DISABLE_DXVA, 1);
            a.SetUINT32(ref HW_TRANSFORMS, 0);
            MFCreateSourceReaderFromURL(path, a, out reader);
            reader.SetStreamSelection(ALL_STREAMS, false);
            reader.SetStreamSelection(FIRST_VIDEO, true);
            IMFMediaType mt;
            MFCreateMediaType(out mt);
            mt.SetGUID(ref MT_MAJOR, ref VIDEO);
            mt.SetGUID(ref MT_SUBTYPE, ref RGB32);
            reader.SetCurrentMediaType(FIRST_VIDEO, IntPtr.Zero, mt);
            Marshal.ReleaseComObject(mt);
            ReadFormat();
        }

        void ReadFormat()
        {
            IMFMediaType cur;
            reader.GetCurrentMediaType(FIRST_VIDEO, out cur);
            ulong size;
            cur.GetUINT64(ref MT_FRAME_SIZE, out size);
            W = (int)(size >> 32);
            H = (int)(size & 0xFFFFFFFF);
            Stride = W * 4;
            try { uint s; cur.GetUINT32(ref MT_DEFAULT_STRIDE, out s); Stride = unchecked((int)s); } catch (COMException) { }
            Marshal.ReleaseComObject(cur);
        }

        // the next decoded frame (held until the next call); false at the end of the clip
        public bool ReadNext()
        {
            if (held != null) { Marshal.ReleaseComObject(held); held = null; }
            while (true)
            {
                uint idx, flags; long ts; IMFSample s;
                reader.ReadSample(FIRST_VIDEO, 0, out idx, out flags, out ts, out s);
                if ((flags & 0x20) != 0) ReadFormat();               // MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED
                if ((flags & 0x1) != 0) throw new Exception("source reader error");
                if ((flags & 0x2) != 0) { if (s != null) Marshal.ReleaseComObject(s); return false; } // end of stream
                if (s == null) continue;                              // a stream tick
                held = s;
                Timestamp = ts;
                return true;
            }
        }

        // the held frame shrunk to dw x dh (box average), BGRX bytes, top row first
        public void Grab(byte[] dst, int dw, int dh)
        {
            IMFMediaBuffer buf;
            held.ConvertToContiguousBuffer(out buf);
            IntPtr p; uint max, cur;
            buf.Lock(out p, out max, out cur);
            try
            {
                int rowBytes = W * 4;
                int stride = Math.Abs(Stride);
                if (frame == null || frame.Length != rowBytes * H) frame = new byte[rowBytes * H];
                for (int r = 0; r < H; r++)
                {
                    long off = Stride >= 0 ? (long)r * stride : (long)(H - 1 - r) * stride;
                    Marshal.Copy(new IntPtr(p.ToInt64() + off), frame, r * rowBytes, rowBytes);
                }
            }
            finally { buf.Unlock(); Marshal.ReleaseComObject(buf); }
            for (int y = 0; y < dh; y++)
            {
                int y0 = (int)((long)y * H / dh), y1 = Math.Max(y0 + 1, (int)((long)(y + 1) * H / dh));
                for (int x = 0; x < dw; x++)
                {
                    int x0 = (int)((long)x * W / dw), x1 = Math.Max(x0 + 1, (int)((long)(x + 1) * W / dw));
                    long b = 0, g = 0, rr = 0; int n = 0;
                    for (int yy = y0; yy < y1; yy += 2)
                    {
                        int row = yy * W * 4;
                        for (int xx = x0; xx < x1; xx += 2)
                        {
                            int i = row + xx * 4;
                            b += frame[i]; g += frame[i + 1]; rr += frame[i + 2]; n++;
                        }
                    }
                    int o = (y * dw + x) * 4;
                    dst[o] = (byte)(b / n); dst[o + 1] = (byte)(g / n); dst[o + 2] = (byte)(rr / n); dst[o + 3] = 255;
                }
            }
        }

        public void Dispose()
        {
            if (held != null) { Marshal.ReleaseComObject(held); held = null; }
            if (reader != null) { Marshal.ReleaseComObject(reader); reader = null; }
            try { MFShutdown(); } catch { }
        }
    }
}
'@

Add-Type -AssemblyName System.Drawing
$r = [TsfClip.Reader]::new((Resolve-Path -LiteralPath $Clip).Path)
try {
    Write-Host ("decoding {0}x{1} (stride {2})" -f $r.W, $r.H, $r.Stride)
    $cw = $CellWidth
    $ch = [int][Math]::Round($cw * $r.H / $r.W)
    $rows = [int][Math]::Ceiling($Per / $Cols)
    $small = New-Object byte[] ($cw * $ch * 4)
    $font = [System.Drawing.Font]::new('Consolas', 13, [System.Drawing.FontStyle]::Bold)
    $back = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(170, 0, 0, 0))
    $ink = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::Yellow)
    $jpeg = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
    $ep = [System.Drawing.Imaging.EncoderParameters]::new(1)
    $ep.Param[0] = [System.Drawing.Imaging.EncoderParameter]::new([System.Drawing.Imaging.Encoder]::Quality, [long]85)
    $sheets = @{}
    foreach ($kind in 'd', 'o') {
        $bmp = [System.Drawing.Bitmap]::new($cw * $Cols, $ch * $rows)
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.Clear([System.Drawing.Color]::Black)
        $sheets[$kind] = @{ Bmp = $bmp; G = $g; N = 1; Used = $false }
    }
    function Save-Sheet($kind) {
        $s = $sheets[$kind]
        if (-not $s.Used) { return }
        $name = '{0}_{1:000}.jpg' -f $kind, $s.N
        $s.Bmp.Save((Join-Path $Out $name), $jpeg, $ep)
        $s.G.Clear([System.Drawing.Color]::Black)
        $s.Used = $false
    }
    function Put-Cell($kind, $sheetNo, $cell, $bmp, $label) {
        $s = $sheets[$kind]
        if ($sheetNo -ne $s.N) { Save-Sheet $kind; $s.N = $sheetNo }
        $x = ($cell % $Cols) * $cw; $y = [Math]::Floor($cell / $Cols) * $ch
        $s.G.DrawImage($bmp, [int]$x, [int]$y, $cw, $ch)
        $sz = $s.G.MeasureString($label, $font)
        $s.G.FillRectangle($back, [float]$x, [float]$y, $sz.Width + 8, $sz.Height + 2)
        $s.G.DrawString($label, $font, $ink, [float]($x + 4), [float]($y + 1))
        $s.Used = $true
    }
    $stepTicks = [long]($Step * 1e7)
    $next = 0L
    $frames = 0; $grabs = 0
    $limit = if ($MaxSeconds -gt 0) { [long]($MaxSeconds * 1e7) } else { [long]::MaxValue }
    $cellBmp = [System.Drawing.Bitmap]::new($cw, $ch, [System.Drawing.Imaging.PixelFormat]::Format32bppRgb)
    $rect = [System.Drawing.Rectangle]::new(0, 0, $cw, $ch)
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($r.ReadNext()) {
        $frames++
        $ts = $r.Timestamp
        if ($ts -gt $limit) { break }
        if ($ts + 80000 -lt $next) { continue }                  # not yet within half a 60 fps frame of the next mark
        $k = [long][Math]::Round($next / $stepTicks)
        $r.Grab($small, $cw, $ch)
        $bd = $cellBmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::WriteOnly, [System.Drawing.Imaging.PixelFormat]::Format32bppRgb)
        [Runtime.InteropServices.Marshal]::Copy($small, 0, $bd.Scan0, $small.Length)
        $cellBmp.UnlockBits($bd)
        $t = $next / 1e7
        Put-Cell 'd' ([Math]::Floor($k / $Per) + 1) ($k % $Per) $cellBmp ('t={0:0.00} #{1}' -f $t, ($k % $Per))
        if ($k % $OverEvery -eq 0) {
            $ok = [long]($k / $OverEvery)
            Put-Cell 'o' ([Math]::Floor($ok / $Per) + 1) ($ok % $Per) $cellBmp ('t={0:0.00} #{1}' -f $t, ($ok % $Per))
        }
        $grabs++
        $next += $stepTicks
        while ($next + 80000 -le $ts) { $next += $stepTicks }     # a gap in the clip: skip the marks it passed
        if ($grabs % 40 -eq 0) { Write-Host ("t={0:0.0} s  frames {1}  grabbed {2}  {3:0} s" -f $t, $frames, $grabs, $watch.Elapsed.TotalSeconds) }
    }
    Save-Sheet 'd'; Save-Sheet 'o'
    Write-Host ("done: {0} frames decoded, {1} grabbed, detail sheets 1..{2}, overview sheets 1..{3}, {4:0} s" -f $frames, $grabs, $sheets['d'].N, $sheets['o'].N, $watch.Elapsed.TotalSeconds)
}
finally {
    $r.Dispose()
}
