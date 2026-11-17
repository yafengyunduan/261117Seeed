"""Read-only runtime resource check. Never creates hardware or downloads files."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'resources/runtime_assets.json').read_text())
    failed = False
    for item in manifest['assets']:
        path = ROOT/item['path']
        if not path.is_file():
            status = 'MISSING'
        else:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            status = 'OK' if digest.hexdigest() == item['sha256'] else 'HASH_MISMATCH'
        print(f"{status:14} {item['path']} ({item['required_for']})")
        failed |= status != 'OK' and item['required_for'] != 'optional upstream demo'
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
