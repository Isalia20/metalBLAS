"""Extract dispatch tables from a captured MPSNDArray Mach-O binary.

    python tools/mps_extract_tables.py /path/to/MPSNDArray --out tables.json

Addresses belong to the exact binary hashes in docs/mps_dispatch_sources.json.
This reads shipped constants; it does not run benchmarks or infer parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import uuid

SOURCES = Path(__file__).resolve().parents[1] / 'docs' / 'mps_dispatch_sources.json'


def extract(binary, sources):
    data = Path(binary).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    matches = [s for s in sources['snapshots'] if s['sha256'] == digest]
    if len(matches) != 1:
        raise ValueError(f'Unknown MPSNDArray SHA-256 {digest}; recover its addresses first')
    source = matches[0]
    if struct.unpack_from('<I', data)[0] != 0xfeedfacf:
        raise ValueError('Expected a little-endian 64-bit Mach-O image')
    pos, segments, image_uuid = 32, [], None
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        command, size = struct.unpack_from('<II', data, pos)
        if command == 0x19:  # LC_SEGMENT_64
            segments.append(struct.unpack_from('<4Q', data, pos + 24))
        elif command == 0x1b:  # LC_UUID
            image_uuid = str(uuid.UUID(bytes=data[pos + 8:pos + 24]))
        pos += size
    if image_uuid != source['uuid']:
        raise ValueError('Mach-O UUID differs from the recorded source')
    tables = {}
    for name, spec in source['tables'].items():
        address = int(spec['address'], 16)
        fmt = '<' + spec['format']
        length = struct.calcsize(fmt) * spec['count']
        for vmaddr, _, fileoff, filesize in segments:
            if vmaddr <= address and address + length <= vmaddr + filesize:
                offset = fileoff + address - vmaddr
                rows = list(struct.iter_unpack(fmt, data[offset:offset + length]))
                tables[name] = [r[0] if len(r) == 1 else list(r) for r in rows]
                break
        else:
            raise ValueError(f'{name} is not contained in a file-backed segment')
    return {'macos': source['macos'], 'build': source['build'], 'uuid': image_uuid,
            'sha256': digest, 'tables': tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.binary, json.loads(SOURCES.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"macOS {result['macos']} ({result['build']}): "
          + ', '.join(f'{k}: {len(v)} rows' for k, v in result['tables'].items()))


if __name__ == '__main__':
    main()
