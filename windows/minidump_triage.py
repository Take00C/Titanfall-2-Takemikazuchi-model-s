"""Read a Windows kernel minidump (PAGEDU64 triage dump) without a debugger.  Read-only.

Prints the bugcheck, the driver whose code the faulting address lies in, the drivers whose
code appears on the crashing thread's stack, and the process that was running.

    python minidump_triage.py <file.dmp> [...]

Layout used (public DUMP_HEADER64 / TRIAGE_DUMP64 structures):
    0x0000  "PAGEDU64" header: 0x38 bugcheck code, 0x40.. four parameters, 0xFA8 system time
    0x2000  triage block: offsets of the call stack, the driver list and its string pool
    driver list entries: u32 name offset, pad, then a loader entry (DllBase +0x30, SizeOfImage +0x40,
    TimeDateStamp +0x80)
"""
import datetime
import re
import struct
import sys

BUGCHECK = {0x3B: "SYSTEM_SERVICE_EXCEPTION", 0x13A: "KERNEL_MODE_HEAP_CORRUPTION", 0x139: "KERNEL_SECURITY_CHECK_FAILURE",
            0x1A: "MEMORY_MANAGEMENT", 0x0A: "IRQL_NOT_LESS_OR_EQUAL", 0xD1: "DRIVER_IRQL_NOT_LESS_OR_EQUAL",
            0x50: "PAGE_FAULT_IN_NONPAGED_AREA", 0x7E: "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED",
            0x1E: "KMODE_EXCEPTION_NOT_HANDLED", 0x9F: "DRIVER_POWER_STATE_FAILURE", 0x133: "DPC_WATCHDOG_VIOLATION",
            0x124: "WHEA_UNCORRECTABLE_ERROR", 0xEF: "CRITICAL_PROCESS_DIED", 0x116: "VIDEO_TDR_FAILURE",
            0x119: "VIDEO_SCHEDULER_INTERNAL_ERROR", 0xC2: "BAD_POOL_CALLER", 0x19: "BAD_POOL_HEADER",
            0x4E: "PFN_LIST_CORRUPT", 0x1000007E: "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED_M", 0x154: "UNEXPECTED_STORE_EXCEPTION",
            0x7F: "UNEXPECTED_KERNEL_MODE_TRAP", 0xC5: "DRIVER_CORRUPTED_EXPOOL", 0x109: "CRITICAL_STRUCTURE_CORRUPTION"}


def u32(d, o):
    return struct.unpack_from("<I", d, o)[0]


def u64(d, o):
    return struct.unpack_from("<Q", d, o)[0]


def drivers(d):
    t = 0x2000
    list_off, count, pool_off, pool_size = u32(d, t + 0x30), u32(d, t + 0x34), u32(d, t + 0x38), u32(d, t + 0x3C)
    if not (0 < count < 1000 and 0 < list_off < len(d)):
        raise ValueError(f"implausible driver list (offset {list_off:#x}, count {count})")
    best = None
    for size in (0x90, 0x88, 0x98, 0xA0, 0x80):                   # entry size varies by Windows build
        mods = []
        for i in range(count):
            e = list_off + i * size
            if e + size > len(d):
                break
            name_off = u32(d, e)
            base, img = u64(d, e + 8 + 0x30), u32(d, e + 8 + 0x40)
            if not (0 < name_off < len(d) - 4):
                break
            n = u32(d, name_off)
            if not (0 < n < 260):
                break
            name = d[name_off + 4:name_off + 4 + 2 * n].decode("utf-16-le", "replace")
            if not re.fullmatch(r"[\x20-\x7e]+", name) or base < 0xFFFF000000000000:
                break
            stamp = u32(d, e + 8 + 0x80)
            mods.append({"name": name, "base": base, "size": img, "stamp": stamp})
        if best is None or len(mods) > len(best):
            best = mods
        if len(mods) == count:
            break
    return best or []


def module_of(mods, addr):
    for m in mods:
        if m["base"] <= addr < m["base"] + m["size"]:
            return m
    return None


def analyse(path):
    d = open(path, "rb").read()
    if d[:8] != b"PAGEDU64":
        print(f"{path}: not a 64-bit kernel minidump")
        return
    code = u32(d, 0x38)
    p = [u64(d, 0x40 + 8 * i) for i in range(4)]
    build = u32(d, 0x0C)
    try:
        when = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=u64(d, 0xFA8) // 10)
        when_s = when.strftime("%Y-%m-%d %H:%M UTC")
    except OverflowError:
        when_s = "?"
    up = u64(d, 0x1030) / 1e7 if u64(d, 0x1030) < 1e15 else None
    print(f"\n=== {path.split(chr(92))[-1]}   crashed {when_s}   Windows build {build}"
          + (f"   uptime {up / 60:.0f} min" if up else ""))
    print(f"  bugcheck {code:#x} {BUGCHECK.get(code, '')}   params " + " ".join(f"{x:#x}" for x in p))
    mods = drivers(d)
    print(f"  {len(mods)} drivers listed in the dump")
    # addresses worth resolving: for 0x3B the 2nd parameter is the faulting instruction
    cand = {"parameter 2 (faulting instruction)" if code == 0x3B else "parameter 2": p[1], "parameter 1": p[0],
            "parameter 3": p[2], "parameter 4": p[3], "exception address": u64(d, 0xF10)}
    for label, a in cand.items():
        m = module_of(mods, a)
        if m:
            print(f"  {label}: {a:#x} = {m['name']} + {a - m['base']:#x}")
    # stack scan: every pointer-sized value on the saved stack that points into a driver
    t = 0x2000
    stack_off, stack_size = u32(d, t + 0x28), u32(d, t + 0x2C)
    seq = []
    if 0 < stack_off < len(d) and 0 < stack_size <= len(d) - stack_off:
        for o in range(stack_off, stack_off + stack_size - 7, 8):
            m = module_of(mods, u64(d, o))
            if m:
                seq.append((m["name"], u64(d, o) - m["base"]))
    order = []
    for name, _off in seq:
        if not order or order[-1][0] != name:
            order.append([name, 1])
        else:
            order[-1][1] += 1
    print("  code addresses on the crashing stack, top first (driver x count):")
    print("    " + "  >  ".join(f"{n} x{c}" for n, c in order[:40]))
    counts = {}
    for name, _ in seq:
        counts[name] = counts.get(name, 0) + 1
    third = {n: c for n, c in counts.items() if not re.match(
        r"(?i)^(ntoskrnl\.exe|hal\.dll|win32k.*|fltmgr\.sys|ntfs\.sys|ci\.dll|clfs\.sys|ksecdd\.sys|cng\.sys|tcpip\.sys|"
        r"ndis\.sys|afd\.sys|netio\.sys|volsnap\.sys|storport\.sys|stornvme\.sys|classpnp\.sys|disk\.sys|partmgr\.sys|"
        r"volmgr.*|mountmgr\.sys|wdf01000\.sys|acpi\.sys|pci\.sys|dxgkrnl\.sys|dxgmms2\.sys|watchdog\.sys|cdd\.dll|"
        r"wdfilter\.sys|fileinfo\.sys|wof\.sys|bindflt\.sys|wcifs\.sys|cldflt\.sys|luafv\.sys|npfs\.sys|msfs\.sys|"
        r"ksecpkg\.sys|fwpkclnt\.sys|pcw\.sys|cea\.sys|werkernel\.sys|kd.*|mcupdate.*|tm\.sys|pshed\.dll|bootvid\.dll)$", n)}
    print("  third-party or less common drivers on that stack:", third or "none")
    # which process: the saved process block holds the image name as ASCII
    proc_off, thread_off = u32(d, t + 0x20), u32(d, t + 0x24)
    if 0 < proc_off < len(d):
        end = thread_off if proc_off < thread_off <= len(d) else min(len(d), proc_off + 0x1000)
        names = sorted({m.group(0).decode() for m in re.finditer(rb"[A-Za-z0-9_\-\.]{2,14}\.(?:exe|EXE)", d[proc_off:end])})
        print("  process running at the time:", names or "not found in the dump")
    return mods


if __name__ == "__main__":
    for f in sys.argv[1:]:
        try:
            analyse(f)
        except Exception as exc:                                   # keep going through the other dumps
            print(f"{f}: could not parse ({type(exc).__name__}: {exc})")
