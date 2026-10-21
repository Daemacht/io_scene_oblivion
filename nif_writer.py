# SPDX-License-Identifier: GPL-3.0-or-later
"""
Schema driven NIF writer (Oblivion 20.0.0.4 / 20.0.0.5, BS stream 11).

Mirror image of :mod:`nif_reader`: blocks are dictionaries of field values keyed
by the field names used in nif.xml.  Missing fields default to 0 / empty, so
builders only need to supply values that matter - but every array length field
must agree with its array (checked while writing).
"""
import struct

import numpy as np

from . import nif_schema as S
from .nif_reader import (Reader, FDict, Block, NifError, _BASIC, _BASIC_SIZE,
                         _STRING_STRUCTS, _WRAPPED_STRING_STRUCTS)


class NifWriteError(Exception):
    pass


class Writer(Reader):
    def __init__(self, version=0x14000005, user=11, bsver=11):
        super().__init__(b'', version, user, bsver)
        self.out = bytearray()

    # ---- primitives -------------------------------------------------------
    def put_basic(self, t, v):
        fmt = _BASIC[t][0]
        if fmt in 'fe':
            v = float(v)
        else:
            v = int(v)
        try:
            self.out += struct.pack('<' + fmt, v)
        except struct.error as e:
            raise NifWriteError('Value %r does not fit %s: %s' % (v, t, e))

    def put_string(self, s, prefix='I'):
        raw = (s or '').encode('latin-1', 'replace')
        self.out += struct.pack('<I', len(raw)) + raw

    def write_value(self, t, val, tpl, arg, arg2=0):
        if t == '#T#':
            t = tpl
        st = self._storage(t)
        if st in _BASIC:
            self.put_basic(st, 0 if val is None else val)
        elif t in S.STRUCTS:
            self.write_struct(t, val, tpl, arg, arg2)
        else:
            raise NifWriteError('Unknown type %r' % t)

    def write_array(self, t, val, n, tpl, arg, arg2=0):
        n = int(n)
        if t == '#T#':
            t = tpl
        if val is None:
            val = [] if n == 0 else None
            if val is None:
                raise NifWriteError('Missing array value for type %s (length %d)' % (t, n))
        st = self._storage(t)
        if t == 'char':
            raw = val.encode('latin-1', 'replace') if isinstance(val, str) else bytes(val)
            if len(raw) != n:
                raise NifWriteError('char array length %d != %d' % (len(raw), n))
            self.out += raw
            return
        if st in _BASIC:
            dt = np.dtype(_BASIC[st][1])
            if t == 'byte' and isinstance(val, (bytes, bytearray)):
                raw = bytes(val)
            else:
                arr = np.asarray(val)
                if t == 'bool':
                    arr = arr.astype(np.uint8)
                raw = np.ascontiguousarray(arr, dtype=dt).tobytes()
            if len(raw) != n * dt.itemsize:
                raise NifWriteError('Array of %s has %d elements, expected %d' % (t, len(raw) // dt.itemsize, n))
            self.out += raw
            return
        flat = self._flat_info(t) if t in S.STRUCTS else None
        if flat is not None and isinstance(val, np.ndarray):
            dt, names = flat
            if val.dtype.names:
                arr = val.astype(dt)
            else:
                base = dt[0]
                arr = np.ascontiguousarray(val, dtype=base).reshape(-1)
                arr = arr.view(dt) if len(dt.names) == 1 else arr.reshape(-1, len(dt.names)).copy().view(
                    np.dtype([(nm, base) for nm in dt.names])).reshape(-1)
            if len(arr) != n:
                raise NifWriteError('Array of %s has %d elements, expected %d' % (t, len(arr), n))
            self.out += arr.astype(dt).tobytes()
            return
        if len(val) != n:
            raise NifWriteError('Array of %s has %d elements, expected %d' % (t, len(val), n))
        for item in val:
            self.write_value(t, item, tpl, arg, arg2)

    def write_struct(self, sname, val, tpl=None, arg=0, arg2=0):
        if sname == 'SizedString':
            self.put_string(val)
            return
        if sname == 'ShortString':
            raw = (val or '').encode('latin-1', 'replace')[:254] + b'\0'
            self.out += bytes([len(raw)]) + raw
            return
        if sname in _WRAPPED_STRING_STRUCTS:
            self.put_string(val if isinstance(val, str) else '')
            return
        F = val if isinstance(val, FDict) else FDict(val or {})
        self._emit_fields(self._struct_fields(sname), F, tpl, arg, arg2)

    def _emit_fields(self, fields, F, tpl, arg, arg2):
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
                ftpl = tpl if f.tpl == '#T#' else f.tpl
                t = tpl if f.type == '#T#' else f.type
                val = F.get(f.name)
                try:
                    if n is None:
                        self.write_value(t, val, ftpl, a, a2)
                    elif f.wid is not None:
                        w = eval(f.wid, env)
                        if isinstance(w, (int, np.integer)):
                            total = int(n) * int(w)
                            if val is None:
                                v2 = None
                            elif isinstance(val, np.ndarray):
                                v2 = val.reshape((total,) + val.shape[2:])
                            elif isinstance(val, (bytes, bytearray)):
                                v2 = val
                            else:
                                v2 = [x for sub in val for x in sub]
                            self.write_array(t, v2, total, ftpl, a, a2)
                        else:
                            if val is None:
                                val = [None] * int(n)
                            for i in range(int(n)):
                                self.write_array(t, val[i], int(w[i]), ftpl, a, a2)
                    else:
                        self.write_array(t, val, n, ftpl, a, a2)
                except NifWriteError as e:
                    raise NifWriteError('field %r: %s' % (f.name, e))
        finally:
            env['F'], env['ARG'], env['ARG1'], env['ARG2'] = saved

    def write_block(self, blk):
        self._emit_fields(self._object_fields(blk.btype), blk, None, 0, 0)


def new_block(btype, **fields):
    b = Block()
    b.btype = btype
    b.index = -1
    b.offset = 0
    b.update(fields)
    return b


def nif_bytes(blocks, roots, version=0x14000005, user=11, bsver=11,
              author='', process_script='', export_script=''):
    """Serialise ``blocks`` (list of Block, indices = list position) to NIF bytes."""
    for i, b in enumerate(blocks):
        b.index = i
    w = Writer(version, user, bsver)
    types, type_index = [], {}
    for b in blocks:
        if b.btype not in type_index:
            type_index[b.btype] = len(types)
            types.append(b.btype)
    hdr = bytearray()
    hdr += b'Gamebryo File Format, Version %d.%d.%d.%d\n' % (
        version >> 24, (version >> 16) & 255, (version >> 8) & 255, version & 255)
    hdr += struct.pack('<IBII', version, 1, user, len(blocks))
    if user >= 3:
        hdr += struct.pack('<I', bsver)
        for s in (author, process_script, export_script):
            raw = s.encode('latin-1', 'replace')[:254] + b'\0'
            hdr += bytes([len(raw)]) + raw
    hdr += struct.pack('<H', len(types))
    for t in types:
        hdr += struct.pack('<I', len(t)) + t.encode('ascii')
    hdr += np.array([type_index[b.btype] for b in blocks], dtype='<u2').tobytes()
    hdr += struct.pack('<I', 0)                     # no groups
    for b in blocks:
        try:
            w.write_block(b)
        except NifWriteError as e:
            raise NifWriteError('Block %d (%s): %s' % (b.index, b.btype, e))
    tail = struct.pack('<I', len(roots)) + b''.join(struct.pack('<i', r.index) for r in roots)
    return bytes(hdr) + bytes(w.out) + tail, len(hdr)


def write_nif(path, blocks, roots, **kw):
    data, _ = nif_bytes(blocks, roots, **kw)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)
