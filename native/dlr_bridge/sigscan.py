"""Byte signatures for the bridge's game addresses, so a game update that only moves code doesn't break it.

    python3 sigscan.py gen     write signatures.h from the current game (the RVAs below are this build's)
    python3 sigscan.py check   find every signature in the installed game's DLLs and print the RVAs

A signature is masked bytes (rel32 and RIP displacements wildcarded) that occur once in the module's .text, and how
to get the address from the match: the match itself plus a delta (functions, return addresses), or the target of a
RIP-relative operand inside it (globals). vtables are found through RTTI names at run time instead (bridge.c).
"""
import os
import re
import struct
import subprocess
import sys

import pefile

GAME = os.path.expanduser('~/.local/share/Steam/steamapps/common/Deadlock/game')
MODULES = {'client': f'{GAME}/citadel/bin/win64/client.dll', 'tier0': f'{GAME}/bin/win64/tier0.dll',
           'server': f'{GAME}/citadel/bin/win64/server.dll'}

# name, module, kind, rva (client 6864 / tier0 of the same date)
#   func: signature at the address; ret: the return address of the call just before it; ref: a global, found
#   through an instruction that references it (the signature covers that instruction)
TARGETS = [
    ('SETUP', 'client', 'func', 0x1a12120),
    ('CALLER', 'client', 'ret', 0x1a0bb44),
    ('SETUP_DELEGATE', 'client', 'func', 0x5ee470),
    ('SKELETON', 'client', 'func', 0x188b500),
    ('ENTITY_LIST', 'client', 'ref', 0x3632e68),
    # Recomputes a scene object's world bounds from its matrix and hands them to the scene system (called right after
    # the game writes the matrix of a moved model; found by trapping writes to a prop's scene object matrix)
    ('SCENE_BOUNDS', 'client', 'func', 0x54f580),
    ('CLIENT_GLOBALS', 'client', 'ref', 0x3471cc0),
    ('CVAR_SET', 'tier0', 'func', 0x20fe50),
    # CCitadelGameRules::BuildGameSessionManifest and the routine adding one resource path to a manifest (found
    # with Modlock's patterns, data/game_signatures.txtpb "game-rules.manifest-builder" / "game-rules.add-resource")
    ('MANIFEST', 'server', 'func', 0x9f2960),
    ('ADD_RESOURCE', 'server', 'func', 0x1f4bbf0),
]
MIN_LEN, MAX_LEN, MARGIN = 16, 96, 8


class Module:
    def __init__(self, path):
        self.path = path
        pe = pefile.PE(path, fast_load=True)
        self.base = pe.OPTIONAL_HEADER.ImageBase
        self.image = pe.get_memory_mapped_image()
        text = next(s for s in pe.sections if s.Name.rstrip(b'\0') == b'.text')
        self.text = (text.VirtualAddress, text.VirtualAddress + text.Misc_VirtualSize)
        self._insns = {}

    def instructions(self, start, length):
        """(rva, size, mask) for the instructions from start: mask 0 on rel32 / RIP displacement bytes."""
        out = subprocess.run(['llvm-objdump', '-d', f'--start-address={self.base + start:#x}',
                              f'--stop-address={self.base + start + length + 16:#x}', self.path],
                             capture_output=True, text=True).stdout
        found = []
        for line in out.splitlines():
            m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)', line)
            if not m:
                continue
            rva = int(m.group(1), 16) - self.base
            raw = bytes.fromhex(m.group(2).replace(' ', ''))
            text = m.group(3)
            mask = [1] * len(raw)
            if raw[0] in (0xe8, 0xe9) or (raw[0] == 0x0f and len(raw) == 6 and 0x80 <= raw[1] <= 0x8f):
                mask[-4:] = [0] * 4
            elif '%rip' in text:
                disp = re.search(r'(-?0x[0-9a-f]+)\(%rip\)', text)
                value = struct.pack('<i', int(disp.group(1), 16)) if disp else None
                at = raw.rfind(value) if value else -1
                if at >= 0:
                    mask[at:at + 4] = [0] * 4
            found.append((rva, len(raw), mask, text))
        return found

    def count(self, pattern, mask, limit=2):
        """Matches of a masked pattern in .text (up to limit)."""
        lo, hi = self.text
        body = bytes(self.image[lo:hi])
        rx = re.compile(b''.join(re.escape(bytes([b])) if k else b'.' for b, k in zip(pattern, mask)), re.S)
        hits = []
        for m in rx.finditer(body):
            hits.append(lo + m.start())
            if len(hits) >= limit:
                break
        return hits


def signature(mod, start, need_cover=0):
    """Shortest whole-instruction signature from start that is unique in .text (plus a margin)."""
    insns = mod.instructions(start, MAX_LEN)
    pattern, mask = b'', []
    unique_at = None
    for rva, size, m, _ in insns:
        pattern += bytes(mod.image[rva:rva + size])
        mask += m
        if len(pattern) < max(MIN_LEN, need_cover):
            continue
        if unique_at is None and len(mod.count(pattern, mask)) == 1:
            unique_at = len(pattern)
        if unique_at is not None and len(pattern) >= unique_at + MARGIN:
            return pattern, mask
        if len(pattern) >= MAX_LEN:
            break
    return (pattern, mask) if unique_at is not None else None


def references(mod, target):
    """RVAs of RIP-relative mov/lea instructions (REX.W 8b/8d, modrm rip) that reference target."""
    lo, hi = mod.text
    body = bytes(mod.image[lo:hi])
    refs = []
    for m in re.finditer(rb'[\x48\x4c][\x8b\x8d][\x05\x0d\x15\x1d\x25\x2d\x35\x3d]', body):
        at = lo + m.start()
        disp = struct.unpack_from('<i', body, m.start() + 3)[0]
        if at + 7 + disp == target:
            refs.append(at)
    return refs


def gen():
    mods = {name: Module(path) for name, path in MODULES.items()}
    lines = ['// Generated by sigscan.py gen from client 6864 / its tier0. kind: 0 = match + delta, 1 = rel32 at',
             '// match + delta (target = its end + rel32). Wildcards are 0 in the mask.', '']
    for name, module, kind, rva in TARGETS:
        mod = mods[module]
        if kind == 'func':
            sig, delta, how = signature(mod, rva), 0, 0
        elif kind == 'ret':
            # The call before the return address: back up to the start of the instruction ending at rva.
            prev = mod.instructions(rva - 16, 16)
            call = next(r for r, s, _, _ in prev if r + s == rva)
            sig, delta, how = signature(mod, call), rva - call, 0
        else:
            sig = None
            for ref in references(mod, rva):
                sig = signature(mod, ref)
                if sig:
                    delta, how = 3, 1
                    break
        if sig is None:
            sys.exit(f'{name}: no unique signature')
        pattern, mask = sig
        lines.append(f'static const u8 kSig{name}[] = {{{", ".join(f"0x{b:02x}" for b in pattern)}}};')
        lines.append(f'static const u8 kSig{name}Mask[] = {{{", ".join(str(k) for k in mask)}}};')
        lines.append(f'#define SIG_{name}_DELTA {delta}')
        lines.append(f'#define SIG_{name}_KIND {how}')
        lines.append('')
        print(name, hex(rva), len(pattern), 'bytes')
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signatures.h'), 'w').write('\n'.join(lines))


def check():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signatures.h')).read()
    mods = {name: Module(path) for name, path in MODULES.items()}
    for name, module, kind, rva in TARGETS:
        pattern = bytes(int(x, 16) for x in re.search(rf'kSig{name}\[\] = {{(.*?)}}', src).group(1).split(', '))
        mask = [int(x) for x in re.search(rf'kSig{name}Mask\[\] = {{(.*?)}}', src).group(1).split(', ')]
        delta = int(re.search(rf'SIG_{name}_DELTA (\d+)', src).group(1))
        how = int(re.search(rf'SIG_{name}_KIND (\d+)', src).group(1))
        hits = mods[module].count(pattern, mask, 3)
        if len(hits) != 1:
            print(name, 'matches', len(hits))
            continue
        at = hits[0] + delta
        found = at + 4 + struct.unpack_from('<i', mods[module].image, at)[0] if how else at
        print(name, hex(found), 'same' if found == rva else f'MOVED from {hex(rva)}')


if __name__ == '__main__':
    {'gen': gen, 'check': check}[sys.argv[1]]()
