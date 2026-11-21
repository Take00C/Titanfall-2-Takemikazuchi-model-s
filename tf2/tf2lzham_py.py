# Pure-Python port of the LZHAM (alpha, Titanfall 2 parameters) *decompressor* only.
# Ported by reading pg9182/tf2lzham cgo/lzham_lzdecomp.cpp, lzham_symbol_codec.{h,cpp},
# lzham_huffman_codes.cpp, lzham_prefix_coding.cpp, lzham_lzdecompbase.cpp.
# Read-only research helper: decompresses VPK chunks in memory, verifies adler32/crc32.
import zlib

NUM_STATES = 12
NUM_LIT_STATES = 7
MAX_MATCH_LEN = 257
MIN_MATCH_LEN = 2
LIT_NEXT_STATE = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10]
HUGE_BASE = [MAX_MATCH_LEN + 1, MAX_MATCH_LEN + 1 + 256, MAX_MATCH_LEN + 1 + 256 + 1024, MAX_MATCH_LEN + 1 + 256 + 1024 + 4096]
HUGE_BITS = [8, 10, 12, 16]
MAX_EXPECTED_CODE_SIZE = 16
MAX_TABLE_BITS = 11


def floor_log2i(v):
    l = 0
    while v > 1:
        v >>= 1
        l += 1
    return l


def ceil_log2i(v):
    l = floor_log2i(v)
    if l != 32 and v > (1 << l):
        l += 1
    return l


def init_position_slots(dict_size_log2):
    extra_bits = [0] * 128
    j = 0
    for i in range(0, 128, 2):
        extra_bits[i] = j
        extra_bits[i + 1] = j
        if i != 0 and j < 25:
            j += 1
    base = [0] * 128
    j = 0
    for i in range(128):
        base[i] = j
        j += (1 << extra_bits[i])
    largest = (1 << dict_size_log2) - 1
    num_slots = 0
    for i in range(128):
        if base[i] <= largest < base[i] + (1 << extra_bits[i]):
            num_slots = i + 1
            break
    assert num_slots
    return num_slots, base, extra_bits


def calculate_minimum_redundancy(A, n):
    if n == 0:
        return
    if n == 1:
        A[0] = 0
        return
    A[0] += A[1]
    root = 0
    leaf = 2
    for nxt in range(1, n - 1):
        if leaf >= n or A[root] < A[leaf]:
            A[nxt] = A[root]
            A[root] = nxt
            root += 1
        else:
            A[nxt] = A[leaf]
            leaf += 1
        if leaf >= n or (root < nxt and A[root] < A[leaf]):
            A[nxt] += A[root]
            A[root] = nxt
            root += 1
        else:
            A[nxt] += A[leaf]
            leaf += 1
    A[n - 2] = 0
    for nxt in range(n - 3, -1, -1):
        A[nxt] = A[A[nxt]] + 1
    avbl = 1
    used = 0
    dpth = 0
    root = n - 2
    nxt = n - 1
    while avbl > 0:
        while root >= 0 and A[root] == dpth:
            used += 1
            root -= 1
        while avbl > used:
            A[nxt] = dpth
            nxt -= 1
            avbl -= 1
        avbl = 2 * used
        dpth += 1
        used = 0


def generate_huffman_codes(freq):
    n = len(freq)
    sizes = [0] * n
    used = [(f, i) for i, f in enumerate(freq) if f]
    total = sum(f for f, _ in used)
    if len(used) == 1:
        sizes[used[0][1]] = 1
        return sizes, 1, total
    used.sort(key=lambda t: t[0])  # stable, same order as the 2-pass LSD radix sort
    x = [f for f, _ in used]
    calculate_minimum_redundancy(x, len(x))
    mx = 0
    for (f, i), l in zip(used, x):
        sizes[i] = l
        if l > mx:
            mx = l
    return sizes, mx, total


def generate_polar_codes(freq):
    n = len(freq)
    sizes = [0] * n
    used = [(f, i) for i, f in enumerate(freq) if f]
    total = sum(f for f, _ in used)
    if len(used) == 1:
        sizes[used[0][1]] = 1
        return sizes, 1, total
    used.sort(key=lambda t: t[0])  # stable ascending, as the radix sort
    m = len(used)
    tmp = [0] * m
    cur_total = 0
    for i in range(m):
        sf = used[m - 1 - i][0]
        adj = 1 << (sf.bit_length() - 1)
        tmp[i] = adj
        cur_total += adj
    tree_total = 1 << (total.bit_length() - 1)
    if tree_total < total:
        tree_total <<= 1
    start_index = 0
    while cur_total < tree_total and start_index < m:
        i = start_index
        while i < m:
            f = tmp[i]
            if cur_total + f <= tree_total:
                tmp[i] += f
                cur_total += f
                if cur_total == tree_total:
                    break
            else:
                start_index = i + 1
            i += 1
    ttb = tree_total.bit_length()
    mx = 0
    for i in range(m):
        cs = ttb - tmp[i].bit_length()
        if cs > mx:
            mx = cs
        sizes[used[m - 1 - i][1]] = cs
    return sizes, mx, total


def limit_max_code_size(sizes, max_code_size):
    cMaxEver = 34
    n = len(sizes)
    num_codes = [0] * (cMaxEver + 1)
    should = False
    for c in sizes:
        num_codes[c] += 1
        if c > max_code_size:
            should = True
    if not should:
        return True
    ofs = 0
    next_sorted_ofs = [0] * (cMaxEver + 1)
    for i in range(1, cMaxEver + 1):
        next_sorted_ofs[i] = ofs
        ofs += num_codes[i]
    if ofs < 2 or ofs > 1024:
        return True
    if ofs > (1 << max_code_size):
        return False
    for i in range(max_code_size + 1, cMaxEver + 1):
        num_codes[max_code_size] += num_codes[i]
    total = 0
    for i in range(max_code_size, 0, -1):
        total += num_codes[i] << (max_code_size - i)
    if total == (1 << max_code_size):
        return True
    while True:
        num_codes[max_code_size] -= 1
        i = max_code_size - 1
        while i:
            if num_codes[i]:
                num_codes[i] -= 1
                num_codes[i + 1] += 2
                break
            i -= 1
        if not i:
            return False
        total -= 1
        if total == (1 << max_code_size):
            break
    new_sizes = []
    for i in range(1, max_code_size + 1):
        new_sizes.extend([i] * num_codes[i])
    for i in range(n):
        c = sizes[i]
        if c:
            o = next_sorted_ofs[c]
            next_sorted_ofs[c] = o + 1
            sizes[i] = new_sizes[o]
    return True


class HuffModel(object):
    __slots__ = ('polar', 'n', 'fast', 'cfg_table_bits', 'max_cycle', 'freq', 'update_cycle', 'total_count',
                 'until_update', 'lookup', 'table_bits', 'table_max_code', 'max_codes', 'val_ptrs',
                 'sorted', 'start_len', 'shift')

    def __init__(self, n=0, fast=False, src=None, polar=False):
        if src is not None:
            self.copy_from(src)
            return
        self.n = n
        self.fast = fast
        self.polar = polar
        self.cfg_table_bits = 0 if n <= 16 else min(1 + ceil_log2i(n), MAX_TABLE_BITS)
        if fast:
            mc = (max(64, n) + 6) << 5
        else:
            mc = (max(24, n) + 6) * 12
        self.max_cycle = min(mc, 32767)
        self.reset()

    def copy_from(self, o):
        self.n = o.n
        self.fast = o.fast
        self.polar = o.polar
        self.cfg_table_bits = o.cfg_table_bits
        self.max_cycle = o.max_cycle
        self.freq = list(o.freq)
        self.update_cycle = o.update_cycle
        self.total_count = o.total_count
        self.until_update = o.until_update
        self.lookup = o.lookup  # immutable after build (rebuilt as new list on update)
        self.table_bits = o.table_bits
        self.table_max_code = o.table_max_code
        self.max_codes = o.max_codes
        self.val_ptrs = o.val_ptrs
        self.sorted = o.sorted
        self.start_len = o.start_len
        self.shift = o.shift

    def reset(self):
        self.freq = [1] * self.n
        self.update_cycle = self.n
        self.total_count = 0
        self.until_update = 0
        self.update()
        self.until_update = self.update_cycle = 8

    def rescale(self):
        f = self.freq
        tot = 0
        for i in range(self.n):
            v = (f[i] + 1) >> 1
            tot += v
            f[i] = v
        self.total_count = tot

    def reset_update_rate(self):
        self.total_count += (self.update_cycle - self.until_update)
        if self.total_count > self.n:
            self.rescale()
        self.until_update = self.update_cycle = min(8, self.update_cycle)

    def update(self):
        self.total_count += self.update_cycle
        while self.total_count >= 32768:
            self.rescale()
        if self.polar:
            sizes, max_code_size, total_freq = generate_polar_codes(self.freq)
        else:
            sizes, max_code_size, total_freq = generate_huffman_codes(self.freq)
        if total_freq != self.total_count:
            raise ValueError('huffman model desync: total_freq %d != total_count %d' % (total_freq, self.total_count))
        if max_code_size > MAX_EXPECTED_CODE_SIZE:
            if not limit_max_code_size(sizes, MAX_EXPECTED_CODE_SIZE):
                raise ValueError('limit_max_code_size failed')
        self.gen_tables(sizes)
        if self.fast:
            self.update_cycle = 2 * self.update_cycle
        else:
            self.update_cycle = (5 * self.update_cycle) >> 2
        if self.update_cycle > self.max_cycle:
            self.update_cycle = self.max_cycle
        self.until_update = self.update_cycle

    def gen_tables(self, sizes):
        n = self.n
        table_bits = self.cfg_table_bits
        num_codes = [0] * (MAX_EXPECTED_CODE_SIZE + 1)
        for c in sizes:
            num_codes[c] += 1
        min_codes = [0] * MAX_EXPECTED_CODE_SIZE
        max_codes = [0] * (MAX_EXPECTED_CODE_SIZE + 1)
        val_ptrs = [0] * (MAX_EXPECTED_CODE_SIZE + 1)
        sorted_positions = [0] * (MAX_EXPECTED_CODE_SIZE + 1)
        next_code = 0
        total_used = 0
        max_cs = 0
        min_cs = 0xFFFFFFFF
        for i in range(1, MAX_EXPECTED_CODE_SIZE + 1):
            k = num_codes[i]
            if not k:
                max_codes[i - 1] = 0
            else:
                if i < min_cs:
                    min_cs = i
                if i > max_cs:
                    max_cs = i
                min_codes[i - 1] = next_code
                mc = next_code + k - 1
                max_codes[i - 1] = 1 + ((mc << (16 - i)) | ((1 << (16 - i)) - 1))
                val_ptrs[i - 1] = total_used
                sorted_positions[i] = total_used
                next_code += k
                total_used += k
            next_code <<= 1
        srt = [0] * total_used
        for i in range(n):
            c = sizes[i]
            if c:
                p = sorted_positions[c]
                sorted_positions[c] = p + 1
                srt[p] = i
        if table_bits <= min_cs:
            table_bits = 0
        lookup = None
        if table_bits:
            lookup = [0xFFFFFFFF] * (1 << table_bits)
            for cs in range(1, table_bits + 1):
                if not num_codes[cs]:
                    continue
                fillsize = table_bits - cs
                fillnum = 1 << fillsize
                min_code = min_codes[cs - 1]
                max_code = (max_codes[cs - 1] - 1) >> (16 - cs)
                vp = val_ptrs[cs - 1]
                for code in range(min_code, max_code + 1):
                    sym = srt[vp + code - min_code]
                    t0 = code << fillsize
                    v = sym | (cs << 16)
                    if fillnum == 1:
                        lookup[t0] = v
                    else:
                        lookup[t0:t0 + fillnum] = [v] * fillnum
        for i in range(MAX_EXPECTED_CODE_SIZE):
            val_ptrs[i] -= min_codes[i]
        table_max_code = 0
        start_len = min_cs
        if table_bits:
            i = table_bits
            while i >= 1:
                if num_codes[i]:
                    table_max_code = max_codes[i - 1]
                    break
                i -= 1
            if i >= 1:
                start_len = table_bits + 1
                for j in range(table_bits + 1, max_cs + 1):
                    if num_codes[j]:
                        start_len = j
                        break
        max_codes[MAX_EXPECTED_CODE_SIZE] = 0xFFFFFFFF
        val_ptrs[MAX_EXPECTED_CODE_SIZE] = 0xFFFFF
        self.lookup = lookup
        self.table_bits = table_bits
        self.shift = 16 - table_bits
        self.table_max_code = table_max_code
        self.max_codes = max_codes
        self.val_ptrs = val_ptrs
        self.sorted = srt
        self.start_len = start_len


class LzhamError(Exception):
    pass


def decompress(src, out_size, dict_size_log2=20, check=True):
    data = bytes(src) + b'\x00' * 16
    num_slots, pos_base, pos_extra_bits = init_position_slots(dict_size_log2)
    out = bytearray(out_size)

    # bit reader state
    buf = 0
    cnt = 0
    pos = 0

    def get_bits(nb):
        nonlocal buf, cnt, pos
        if nb == 0:
            return 0
        while cnt < nb:
            buf = (buf << 8) | data[pos]
            pos += 1
            cnt += 8
        cnt -= nb
        r = buf >> cnt
        buf &= (1 << cnt) - 1
        return r

    def align():
        nonlocal buf, cnt
        k = cnt & 7
        if k:
            get_bits(k)

    def huff(m):
        nonlocal buf, cnt, pos
        if cnt < 16:
            buf = (buf << 32) | int.from_bytes(data[pos:pos + 4], 'big')
            pos += 4
            cnt += 32
        top16 = buf >> (cnt - 16)
        k = top16 + 1
        if k <= m.table_max_code:
            t = m.lookup[top16 >> m.shift]
            sym = t & 0xFFFF
            ln = t >> 16
        else:
            ln = m.start_len
            mc = m.max_codes
            while k > mc[ln - 1]:
                ln += 1
            if ln > 16:
                vp = 0
            else:
                vp = m.val_ptrs[ln - 1] + (top16 >> (16 - ln))
                if vp < 0 or vp >= m.n:
                    vp = 0
            sym = m.sorted[vp]
        cnt -= ln
        buf &= (1 << cnt) - 1
        m.freq[sym] += 1
        u = m.until_update - 1
        m.until_update = u
        if u == 0:
            m.update()
        return sym

    tmp = get_bits(2)
    fast = (tmp & 2) != 0
    polar = (tmp & 1) != 0

    def new_models():
        lit0 = HuffModel(256, fast, polar=polar)
        lit = [lit0] + [HuffModel(src=lit0) for _ in range(63)]
        dl0 = HuffModel(256, fast, polar=polar)
        dlit = [dl0] + [HuffModel(src=dl0) for _ in range(63)]
        main = HuffModel(2 + (num_slots - 1) * 8, fast, polar=polar)
        rep_len = [HuffModel(1 + (MAX_MATCH_LEN - MIN_MATCH_LEN + 1), fast, polar=polar) for _ in range(2)]
        large_len = [HuffModel(1 + 249, fast, polar=polar) for _ in range(2)]
        dist_lsb = HuffModel(16, fast, polar=polar)
        return lit, dlit, main, rep_len, large_len, dist_lsb

    lit, dlit, main, rep_len, large_len, dist_lsb = new_models()
    is_match = [1024] * (NUM_STATES * 64)
    is_rep = [1024] * NUM_STATES
    is_rep0 = [1024] * NUM_STATES
    is_rep0_single = [1024] * NUM_STATES
    is_rep1 = [1024] * NUM_STATES
    is_rep2 = [1024] * NUM_STATES

    def reset_all_tables():
        nonlocal lit, dlit, main, rep_len, large_len, dist_lsb
        lit, dlit, main, rep_len, large_len, dist_lsb = new_models()
        for arr in (is_match, is_rep, is_rep0, is_rep0_single, is_rep1, is_rep2):
            for i in range(len(arr)):
                arr[i] = 1024

    def reset_update_rates():
        for m in lit:
            m.reset_update_rate()
        for m in dlit:
            m.reset_update_rate()
        main.reset_update_rate()
        for m in rep_len:
            m.reset_update_rate()
        for m in large_len:
            m.reset_update_rate()
        dist_lsb.reset_update_rate()

    arith_value = 0
    arith_length = 0

    def abit(arr, idx):
        nonlocal arith_value, arith_length
        while arith_length < 0x1000000:
            arith_value = ((arith_value << 8) | get_bits(8)) & 0xFFFFFFFF
            arith_length = (arith_length << 8) & 0xFFFFFFFF
        p = arr[idx]
        x = p * (arith_length >> 11)
        if arith_value >= x:
            arr[idx] = p - (p >> 5)
            arith_value -= x
            arith_length -= x
            return 1
        arr[idx] = p + ((2048 - p) >> 5)
        arith_length = x
        return 0

    def huge_len():
        ml = 0
        while True:
            b = get_bits(1)
            if not b:
                break
            ml += 1
            if ml >= 3:
                break
        k = get_bits(HUGE_BITS[ml])
        return HUGE_BASE[ml] + k

    dst = 0
    status = None
    while status is None:
        block_type = get_bits(2)
        if block_type == 0:  # sync
            t = get_bits(2)
            if t == 1:
                reset_update_rates()
            elif t == 2:
                reset_all_tables()
            align()
            if get_bits(16) != 0 or get_bits(16) != 0xFFFF:
                raise LzhamError('bad sync block')
            if t == 2:
                raise LzhamError('full-flush sync block not supported in this port')
        elif block_type == 2:  # raw
            nraw = get_bits(24)
            chk = get_bits(8)
            if chk != ((nraw & 0xFF) ^ ((nraw >> 8) & 0xFF) ^ ((nraw >> 16) & 0xFF)):
                raise LzhamError('bad raw block')
            nraw += 1
            align()
            bytepos = pos - (cnt >> 3)
            if dst + nraw > out_size:
                raise LzhamError('dest too small (raw)')
            out[dst:dst + nraw] = data[bytepos:bytepos + nraw]
            dst += nraw
            pos = bytepos + nraw
            buf = 0
            cnt = 0
        elif block_type == 1:  # comp
            arith_value = get_bits(8)
            arith_value = (arith_value << 8) | get_bits(8)
            arith_value = (arith_value << 8) | get_bits(8)
            arith_value = (arith_value << 8) | get_bits(8)
            arith_length = 0xFFFFFFFF
            h0 = h1 = h2 = h3 = 1
            state = 0
            prev = 0
            pprev = 0
            ft = get_bits(2)
            if ft == 1:
                reset_update_rates()
            elif ft == 2:
                reset_all_tables()
            while True:
                if not abit(is_match, (prev >> 2) + (state << 6)):
                    if dst >= out_size:
                        raise LzhamError('dest too small (lit)')
                    if state < NUM_LIT_STATES:
                        r = huff(lit[(prev >> 5) | ((pprev >> 5) << 3)])
                    else:
                        o = dst - h0
                        rl0 = out[o]
                        rl1 = out[o - 1]
                        r = huff(dlit[(rl0 >> 5) | ((rl1 >> 5) << 3)]) ^ rl0
                    out[dst] = r
                    pprev = prev
                    prev = r
                    state = LIT_NEXT_STATE[state]
                    dst += 1
                else:
                    mlen = 1
                    if abit(is_rep, state):
                        if abit(is_rep0, state):
                            if abit(is_rep0_single, state):
                                state = 9 if state < NUM_LIT_STATES else 11
                            else:
                                mlen = huff(rep_len[1 if state >= NUM_LIT_STATES else 0]) + MIN_MATCH_LEN
                                if mlen == MAX_MATCH_LEN + 1:
                                    mlen = huge_len()
                                state = 8 if state < NUM_LIT_STATES else 11
                        else:
                            mlen = huff(rep_len[1 if state >= NUM_LIT_STATES else 0]) + MIN_MATCH_LEN
                            if mlen == MAX_MATCH_LEN + 1:
                                mlen = huge_len()
                            if abit(is_rep1, state):
                                h0, h1 = h1, h0
                            else:
                                if abit(is_rep2, state):
                                    h0, h1, h2 = h2, h0, h1
                                else:
                                    h0, h1, h2, h3 = h3, h0, h1, h2
                            state = 8 if state < NUM_LIT_STATES else 11
                    else:
                        sym = huff(main) - 2
                        if sym < 0:
                            if sym == -2:
                                break  # end of block
                            h0 = h1 = h2 = h3 = 1
                            state = 0
                            continue
                        mlen = (sym & 7) + 2
                        slot = (sym >> 3) + 1
                        if mlen == 9:
                            mlen += huff(large_len[1 if state >= NUM_LIT_STATES else 0])
                            if mlen == MAX_MATCH_LEN + 1:
                                mlen = huge_len()
                        neb = pos_extra_bits[slot]
                        if neb < 3:
                            eb = get_bits(neb)
                        else:
                            eb = 0
                            if neb > 4:
                                eb = get_bits(neb - 4) << 4
                            eb += huff(dist_lsb)
                        h3 = h2
                        h2 = h1
                        h1 = h0
                        h0 = pos_base[slot] + eb
                        state = NUM_LIT_STATES if state < NUM_LIT_STATES else NUM_LIT_STATES + 3
                    if h0 > dst or dst + mlen > out_size:
                        raise LzhamError('bad code (match dist %d at %d len %d)' % (h0, dst, mlen))
                    s = dst - h0
                    if mlen == 1:
                        c = out[s]
                        out[dst] = c
                        pprev = prev
                        prev = c
                        dst += 1
                    else:
                        if h0 >= mlen:
                            out[dst:dst + mlen] = out[s:s + mlen]
                        elif h0 == 1:
                            out[dst:dst + mlen] = bytes([out[s]]) * mlen
                        else:
                            pat = bytes(out[s:dst])
                            reps = mlen // h0 + 1
                            out[dst:dst + mlen] = (pat * reps)[:mlen]
                        dst += mlen
                        prev = out[dst - 1]
                        pprev = out[dst - 2]
            align()
        else:  # EOF
            status = 'ok'
    align()
    a = get_bits(16)
    a = (a << 16) | get_bits(16)
    if dst != out_size:
        raise LzhamError('size mismatch: got %d expected %d' % (dst, out_size))
    if check:
        mine = zlib.adler32(bytes(out)) & 0xFFFFFFFF
        if mine != a:
            raise LzhamError('adler32 mismatch: stream %08x computed %08x' % (a, mine))
    return bytes(out)
