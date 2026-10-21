# SPDX-License-Identifier: GPL-3.0-or-later
"""
Schema driven NIF reader (Oblivion: NIF 20.0.0.4 / 20.0.0.5, BS stream version 11).

Pre-Skyrim NIF files have no per-block size table, so *every* block has to be
parsed exactly to find the start of the next one.  Rather than hand coding
hundreds of block layouts, this reader interprets the machine readable
description in ``nif_schema.py`` (generated from niftools' nif.xml).

Public API:
    nif = read_nif(path)            # -> NifFile
    nif.blocks                      # list[Block]  (Block is a dict of field name -> value)
    nif.roots                       # list[Block]
    nif.deref(ref)                  # Ref/Ptr int -> Block or None
"""
import struct
from collections import OrderedDict

import numpy as np

from . import nif_schema as S


class NifError(Exception):
    pass


# --------------------------------------------------------------- basic types
# name -> (struct format, numpy dtype)
_BASIC = {
    'byte': ('B', '<u1'), 'sbyte': ('b', '<i1'), 'char': ('B', '<u1'),
    'short': ('h', '<i2'), 'ushort': ('H', '<u2'),
    'int': ('i', '<i4'), 'uint': ('I', '<u4'), 'ulittle32': ('I', '<u4'),
    'int64': ('q', '<i8'), 'uint64': ('Q', '<u8'),
    'float': ('f', '<f4'), 'hfloat': ('e', '<f2'),
    'bool': ('B', '<u1'), 'normbyte': ('B', '<u1'),
    'Ptr': ('i', '<i4'), 'Ref': ('i', '<i4'),
    'StringOffset': ('I', '<u4'), 'NiFixedString': ('I', '<u4'),
    'BlockTypeIndex': ('h', '<i2'), 'FileVersion': ('I', '<u4'),
}
_BASIC_SIZE = {k: struct.calcsize('<' + v[0]) for k, v in _BASIC.items()}

_STRING_STRUCTS = ('SizedString', 'ShortString')      # -> str
_WRAPPED_STRING_STRUCTS = ('string', 'FilePath')      # -> str via inner 'String'


class FDict(dict):
    """Field dictionary; unknown names evaluate to 0 inside conditions."""
    __slots__ = ()

    def __missing__(self, key):
        return 0


class Block(FDict):
    """A parsed NIF block.  Field values are accessed like ``block['Name']``."""
    __slots__ = ('btype', 'index', 'offset')

    def __repr__(self):
        return '<Block %d %s %r>' % (self.index, self.btype, self.get('Name', ''))

    def is_a(self, *names):
        return any(_inherits(self.btype, n) for n in names)


def _inherits(tname, base):
    while tname:
        if tname == base:
            return True
        tname = S.OBJECTS.get(tname, {}).get('inherit')
    return False


# -------------------------------------------------------------- field prep
class _Field:
    __slots__ = ('name', 'type', 'tpl', 'arg', 'arg2', 'len', 'wid', 'cond', 'kind', 'size')


def _comp(expr):
    return compile(expr, '<nif>', 'eval') if expr is not None else None


class Reader:
    def __init__(self, data, version, user, bsver):
        self.data = data
        self.pos = 0
        self.version = version
        self.user = user
        self.bsver = bsver
        self.env = {'__builtins__': {}, 'VER': version, 'USER': user, 'BSVER': bsver,
                    'F': None, 'ARG': 0, 'ARG1': 0, 'ARG2': 0}
        self._obj_cache = {}
        self._struct_cache = {}
        self._flat_cache = {}

    # ---- schema preparation (static filtering by version) ---------------
    def _static_ok(self, f, btype):
        v = self.version
        if 'since' in f and v < f['since']:
            return False
        if 'until' in f and v > f['until']:
            return False
        if 'onlyT' in f and not (btype and _inherits(btype, f['onlyT'])):
            return False
        if 'excludeT' in f and btype and _inherits(btype, f['excludeT']):
            return False
        return True

    def _prep_fields(self, raw_fields, btype):
        out = []
        for f in raw_fields:
            if not self._static_ok(f, btype):
                continue
            cond = f.get('cond')
            vc = f.get('vercond')
            if vc is not None:
                if "F[" in vc:        # depends on fields -> evaluate at runtime
                    cond = '(%s) and (%s)' % (cond, vc) if cond else vc
                else:
                    if not eval(vc, {'__builtins__': {}, 'VER': self.version, 'USER': self.user,
                                     'BSVER': self.bsver}):
                        continue
            pf = _Field()
            pf.name = f['n']
            pf.type = f['t']
            pf.tpl = f.get('tpl')
            pf.arg = _comp(f.get('arg'))
            pf.arg2 = _comp(f.get('arg2'))
            pf.len = _comp(f.get('len'))
            pf.wid = _comp(f.get('wid'))
            pf.cond = _comp(cond)
            out.append(pf)
        return out

    def _object_fields(self, btype):
        fl = self._obj_cache.get(btype)
        if fl is None:
            chain = []
            t = btype
            while t:
                info = S.OBJECTS.get(t)
                if info is None:
                    raise NifError('Unknown block type %r' % t)
                chain.append(info)
                t = info.get('inherit')
            raw = []
            for info in reversed(chain):
                raw.extend(info['fields'])
            fl = self._prep_fields(raw, btype)
            self._obj_cache[btype] = fl
        return fl

    def _struct_fields(self, sname):
        fl = self._struct_cache.get(sname)
        if fl is None:
            info = S.STRUCTS.get(sname)
            if info is None:
                raise NifError('Unknown struct type %r' % sname)
            fl = self._prep_fields(info['fields'], None)
            self._struct_cache[sname] = fl
        return fl

    def _flat_info(self, sname):
        """If a struct is a plain run of numeric fields, return (numpy dtype, field names)."""
        if sname in self._flat_cache:
            return self._flat_cache[sname]
        res = None
        if sname not in _STRING_STRUCTS and sname not in _WRAPPED_STRING_STRUCTS:
            fl = self._struct_fields(sname)
            ok = bool(fl)
            dts = []
            for f in fl:
                if f.cond is not None or f.len is not None or f.wid is not None or f.arg is not None:
                    ok = False
                    break
                t = self._storage(f.type)
                if t not in _BASIC or f.type in ('bool',) and False:
                    ok = False
                    break
                dts.append((f.name, _BASIC[t][1]))
            if ok:
                res = (np.dtype(dts), [f.name for f in fl])
        self._flat_cache[sname] = res
        return res

    @staticmethod
    def _storage(t):
        return S.ENUMS.get(t, t)

    # ---- primitive reads --------------------------------------------------
    def _unpack(self, fmt, size):
        p = self.pos
        if p + size > len(self.data):
            raise NifError('Unexpected end of file at %d' % p)
        self.pos = p + size
        return struct.unpack_from('<' + fmt, self.data, p)[0]

    def read_basic(self, t):
        fmt, _dt = _BASIC[t]
        v = self._unpack(fmt, _BASIC_SIZE[t])
        if t == 'bool':
            return bool(v)
        return v

    def read_array(self, t, n, tpl, arg, arg2=0):
        """Read ``n`` elements of type ``t``."""
        if n < 0 or n > 50_000_000:
            raise NifError('Implausible array length %d at %d' % (n, self.pos))
        st = self._storage(t)
        if t == 'char':
            raw = self.data[self.pos:self.pos + n]
            self.pos += n
            return raw.split(b'\0', 1)[0].decode('latin-1')
        if st in _BASIC:
            dt = np.dtype(_BASIC[st][1])
            if t == 'byte':
                nbytes = n
                if self.pos + nbytes > len(self.data):
                    raise NifError('Unexpected end of file reading bytes')
                raw = bytes(self.data[self.pos:self.pos + nbytes])
                self.pos += nbytes
                return raw
            nbytes = n * dt.itemsize
            if self.pos + nbytes > len(self.data):
                raise NifError('Unexpected end of file reading array at %d' % self.pos)
            arr = np.frombuffer(self.data, dtype=dt, count=n, offset=self.pos)
            self.pos += nbytes
            if t == 'bool':
                arr = arr.astype(bool)
            return arr
        if t == '#T#':
            t = tpl
            return self.read_array(t, n, None, arg, arg2)
        flat = self._flat_info(t) if t in S.STRUCTS else None
        if flat is not None and t not in _STRING_STRUCTS:
            dt, names = flat
            nbytes = n * dt.itemsize
            if self.pos + nbytes > len(self.data):
                raise NifError('Unexpected end of file reading %s array at %d' % (t, self.pos))
            arr = np.frombuffer(self.data, dtype=dt, count=n, offset=self.pos)
            self.pos += nbytes
            if len(dt.names) > 1 and len({dt[i] for i in range(len(dt.names))}) == 1:
                arr = arr.view(dt[0]).reshape(n, len(dt.names))
            return arr
        return [self.read_value(t, tpl, arg, arg2) for _ in range(n)]

    def read_value(self, t, tpl, arg, arg2=0):
        if t == '#T#':
            t = tpl
        st = self._storage(t)
        if st in _BASIC:
            return self.read_basic(st if t != 'bool' else 'bool')
        if t in S.STRUCTS:
            return self.read_struct(t, tpl, arg, arg2)
        raise NifError('Unknown type %r' % t)

    # ---- structured reads -------------------------------------------------
    def _run_fields(self, fields, F, tpl, arg, arg2):
        env = self.env
        saved = (env['F'], env['ARG'], env['ARG1'], env['ARG2'])
        env['F'] = F
        env['ARG'] = arg
        env['ARG1'] = arg
        env['ARG2'] = arg2
        try:
            for f in fields:
                if f.cond is not None and not eval(f.cond, env):
                    continue
                n = eval(f.len, env) if f.len is not None else None
                a = eval(f.arg, env) if f.arg is not None else 0
                a2 = eval(f.arg2, env) if f.arg2 is not None else 0
                ftpl = f.tpl
                if ftpl == '#T#':
                    ftpl = tpl
                t = f.type
                if t == '#T#':
                    t = tpl
                if n is None:
                    val = self.read_value(t, ftpl, a, a2)
                elif f.wid is not None:
                    w = eval(f.wid, env)
                    if isinstance(w, (int, np.integer)):
                        total = int(n) * int(w)
                        arr = self.read_array(t, total, ftpl, a, a2)
                        if isinstance(arr, (np.ndarray,)):
                            val = arr.reshape((int(n), int(w)) + arr.shape[1:])
                        elif isinstance(arr, bytes):
                            val = np.frombuffer(arr, dtype=np.uint8).reshape(int(n), int(w))
                        else:
                            val = [arr[i * w:(i + 1) * w] for i in range(int(n))]
                    else:           # jagged: width is an array indexed by the outer index
                        val = [self.read_array(t, int(w[i]), ftpl, a, a2) for i in range(int(n))]
                else:
                    val = self.read_array(t, int(n), ftpl, a, a2)
                F[f.name] = val
                if env['F'] is not F:       # defensive
                    env['F'] = F
        finally:
            env['F'], env['ARG'], env['ARG1'], env['ARG2'] = saved
        return F

    def read_struct(self, sname, tpl=None, arg=0, arg2=0):
        if sname == 'SizedString':
            n = self._unpack('I', 4)
            if n > 100000:
                raise NifError('Bad string length %d' % n)
            raw = self.data[self.pos:self.pos + n]
            self.pos += n
            return raw.decode('latin-1')
        if sname == 'ShortString':
            n = self._unpack('B', 1)
            raw = self.data[self.pos:self.pos + n]
            self.pos += n
            return raw.split(b'\0', 1)[0].decode('latin-1')
        F = FDict()
        self._run_fields(self._struct_fields(sname), F, tpl, arg, arg2)
        if sname in _WRAPPED_STRING_STRUCTS:
            s = F.get('String')
            return s if isinstance(s, str) else ''
        return F

    def read_block(self, btype, index):
        b = Block()
        b.btype = btype
        b.index = index
        b.offset = self.pos
        self._run_fields(self._object_fields(btype), b, None, 0, 0)
        return b


# ------------------------------------------------------------------ the file
class NifFile:
    def __init__(self):
        self.version = 0
        self.user_version = 0
        self.bs_version = 0
        self.blocks = []
        self.roots = []
        self.block_types = []
        self.warnings = []
        self.header_strings = []
        self.complete = False
        self.blocks_offset = 0

    def deref(self, ref):
        if ref is None:
            return None
        ref = int(ref)
        if 0 <= ref < len(self.blocks):
            return self.blocks[ref]
        return None

    def of_type(self, *names):
        return [b for b in self.blocks if b.is_a(*names)]

    @property
    def version_string(self):
        v = self.version
        return '%d.%d.%d.%d' % (v >> 24, (v >> 16) & 255, (v >> 8) & 255, v & 255)


def read_nif(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, 'rb') as fh:
            data = fh.read()
    return parse_nif(data)


def parse_nif(data):
    nif = NifFile()
    nl = data.find(b'\n', 0, 128)
    if nl < 0 or not (data.startswith(b'Gamebryo File Format') or data.startswith(b'NetImmerse File Format')):
        raise NifError('Not a NIF file (bad header string)')
    p = nl + 1

    def u32():
        nonlocal p
        v = struct.unpack_from('<I', data, p)[0]
        p += 4
        return v

    ver = u32()
    nif.version = ver
    if ver not in (0x14000004, 0x14000005):
        raise NifError('Unsupported NIF version %s: this importer handles Oblivion files '
                       '(20.0.0.4 / 20.0.0.5) only' % NifFile.version_string.fget(nif))
    endian = data[p]
    p += 1
    if endian != 1:
        raise NifError('Big-endian NIF files are not supported')
    user = u32()
    nblocks = u32()
    nif.user_version = user
    bsver = 0
    if user >= 3:                      # BS stream header
        bsver = u32()
        for _ in range(3):             # author, process script, export script (ShortString)
            n = data[p]
            p += 1 + n
    nif.bs_version = bsver
    ntypes = struct.unpack_from('<H', data, p)[0]
    p += 2
    types = []
    for _ in range(ntypes):
        n = u32()
        types.append(data[p:p + n].decode('latin-1'))
        p += n
    idx = np.frombuffer(data, dtype='<u2', count=nblocks, offset=p)
    p += 2 * nblocks
    ngroups = u32()
    p += 4 * ngroups
    nif.block_types = [types[i] for i in idx]
    nif.blocks_offset = p

    rd = Reader(data, ver, user, bsver)
    rd.pos = p
    for i, bt in enumerate(nif.block_types):
        try:
            blk = rd.read_block(bt, i)
        except Exception as e:                          # noqa: BLE001 - report & keep what we have
            nif.warnings.append('Block %d (%s) could not be parsed at byte %d: %s. '
                                'Remaining blocks were skipped.' % (i, bt, rd.pos, e))
            break
        nif.blocks.append(blk)
    else:
        try:
            nroots = rd._unpack('I', 4)
            roots = [rd._unpack('i', 4) for _ in range(nroots)]
            nif.roots = [nif.blocks[r] for r in roots if 0 <= r < len(nif.blocks)]
            nif.complete = True
            if rd.pos != len(data):
                nif.warnings.append('%d unread trailing bytes' % (len(data) - rd.pos))
        except NifError as e:
            nif.warnings.append('Footer unreadable: %s' % e)
    if not nif.roots and nif.blocks:
        nif.roots = [nif.blocks[0]]
    return nif
