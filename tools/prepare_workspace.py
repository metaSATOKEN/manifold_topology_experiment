"""Restore verified saved evidence to a NEW workspace; never runs model/analysis code."""
import argparse
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode = True
from verify_package import ROOT, contained, sha, verify

METADATA = {'ARCHIVE.json', 'ARCHIVE_INDEX.json', 'BUNDLE_INDEX.json'}


def member_target(item, info):
    name = info.filename
    mode = info.external_attr >> 16
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or stat.S_ISLNK(mode):
        raise ValueError('Unsafe archive member: ' + name)
    if info.is_dir():
        return None
    if len(path.parts) == 1 and name in METADATA:
        return Path('_archive_metadata') / item['name'] / name
    prefix = item['member_prefix']
    if prefix:
        if not name.startswith(prefix):
            raise ValueError('Unexpected archive prefix: ' + name)
        name = name[len(prefix):]
    return Path(item['restore_to']) / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Verify and inspect members without extracting.')
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.exists() or dest.is_relative_to(ROOT):
        raise ValueError('Choose a nonexistent destination outside the repository.')
    check = verify(args.assets_dir)
    if check['status'] != 'PASS':
        raise ValueError('Integrity verification failed: ' + json.dumps(check))
    assets = json.loads((ROOT / 'references/ASSETS.json').read_text())['assets']
    count = total = 0
    for item in assets:
        with zipfile.ZipFile(contained(args.assets_dir, item['name'])) as archive:
            for info in archive.infolist():
                target = member_target(item, info)
                if target is not None:
                    contained(dest, str(target))
                    count += 1
                    total += info.file_size
    if args.dry_run:
        print(json.dumps({'status': 'PASS', 'assets': len(assets), 'members': count,
                          'uncompressed_bytes_with_repeated_metadata': total, 'extracted': False}))
        return
    # All destinations and asset hashes were checked before creating any output.
    dest.mkdir(parents=True)
    shutil.copytree(ROOT / 'research', dest / 'research')
    copied = duplicates = 0
    for item in assets:
        with zipfile.ZipFile(contained(args.assets_dir, item['name'])) as archive:
            for info in archive.infolist():
                rel = member_target(item, info)
                if rel is None:
                    continue
                target = contained(dest, str(rel))
                if target.exists():
                    # P4 repeats run.json across independent PART ZIPs.
                    with archive.open(info) as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if sha(target) != digest:
                        raise ValueError('Conflicting repeated member: ' + str(rel))
                    duplicates += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output)
                copied += 1
    result = {'status': 'PASS', 'assets': len(assets), 'files_restored': copied,
              'identical_repeated_members': duplicates, 'model_forwards': 0,
              'analysis_executions': 0, 'scope': 'Offline file restoration only.'}
    (dest / 'RESTORE_REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
