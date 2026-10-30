"""Offline hash verification. Does not execute experiment code or write results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes its directory: ' + name)
    return path


def verify(assets_dir=None):
    manifest = json.loads((ROOT / 'references/MANIFEST.json').read_text())
    errors = []
    for name, info in manifest['files'].items():
        path = contained(ROOT, name)
        if not path.is_file() or path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            errors.append(name)
    assets = json.loads((ROOT / 'references/ASSETS.json').read_text())['assets']
    if assets_dir is not None:
        for item in assets:
            path = contained(assets_dir, item['name'])
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                errors.append('asset:' + item['name'])
    return {'status': 'PASS' if not errors else 'FAIL', 'repository_files': len(manifest['files']),
            'assets_checked': len(assets) if assets_dir is not None else 0,
            'errors': errors, 'scope': 'File integrity only; no experiment or numerical reanalysis.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    report = verify(args.assets_dir)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
