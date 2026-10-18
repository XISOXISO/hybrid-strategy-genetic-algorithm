import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


DATASET_ID = '0a587c6579467ae7d369662d923bb56bc60f53462b2ad9aac3d403523d198439'
EXPECTED_COUNTS = {'train': 3617, 'validation': 776, 'test': 779}
SPEC_FILES = ('dataset.json', 'sample_manifest.csv', 'splits.csv',
              'class_mapping.json', 'split_summary.csv', 'provenance.json')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.pgm', '.ppm'}


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def safe_path(root, relative):
    root = Path(root).resolve()
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    result = (root / part).resolve()
    if root not in result.parents:
        raise ValueError(f'Path escapes the output directory: {relative}')
    return result


def load_spec(folder):
    folder = Path(folder).resolve()
    data = read_json(folder / 'dataset.json')
    identity = data['identity']
    if data['dataset_id'] != DATASET_ID or digest(identity) != DATASET_ID:
        raise ValueError('The frozen experiment dataset identity has changed')
    classes = identity['classes']
    samples = identity['samples']
    if len(classes) != 10 or len(set(classes)) != 10 or len(samples) != 5172:
        raise ValueError('Unexpected class list or sample count')
    counts = Counter(row['split'] for row in samples)
    if dict(counts) != EXPECTED_COUNTS or identity['split_counts'] != EXPECTED_COUNTS:
        raise ValueError('The fixed split counts have changed')
    if len({row['copied_image'] for row in samples}) != len(samples):
        raise ValueError('A sample occurs more than once in the frozen manifest')
    manifest = read_csv(folder / 'sample_manifest.csv')
    splits = read_csv(folder / 'splits.csv')
    if len(manifest) != len(samples) or len(splits) != len(samples):
        raise ValueError('CSV records do not match the frozen manifest')
    positions = Counter()
    expected_summary = Counter()
    for index, (sample, item, split) in enumerate(zip(samples, manifest, splits), 1):
        sid = f'MSTAR_{index:05d}'
        relative = sample['copied_image']
        label, role = sample['label'], sample['split']
        if type(label) is not int or not 0 <= label < len(classes):
            raise ValueError(f'Invalid label at {sid}')
        if sample['class'] != classes[label]:
            raise ValueError(f'Class mismatch at {sid}')
        safe_path(folder, relative)
        expected = {'sample_id': sid, 'relative_path': relative, 'label': str(label),
                    'class_name': sample['class']}
        if any(item.get(k) != v or split.get(k) != v for k, v in expected.items()):
            raise ValueError(f'CSV membership or order mismatch at {sid}')
        if item['filename'] != Path(relative).name:
            raise ValueError(f'Filename mismatch at {sid}')
        if int(item['width']) <= 0 or int(item['height']) <= 0:
            raise ValueError(f'Invalid image dimensions at {sid}')
        for key in ('sha256', 'decoded_sha256'):
            checksum = sample[key]
            if item[key] != checksum or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError(f'Checksum metadata mismatch at {sid}')
        if split['split'] != role or int(split['split_index']) != positions[role]:
            raise ValueError(f'Fixed split order mismatch at {sid}')
        positions[role] += 1
        expected_summary[label, role] += 1
    mapping = read_json(folder / 'class_mapping.json')
    if mapping != {str(i): name for i, name in enumerate(classes)}:
        raise ValueError('Class mapping differs from the frozen manifest')
    summary = read_csv(folder / 'split_summary.csv')
    if len(summary) != len(classes):
        raise ValueError('Invalid class summary')
    for label, row in enumerate(summary):
        if int(row['label']) != label or row['class_name'] != classes[label]:
            raise ValueError('Invalid class summary labels')
        expected_total = 0
        for role in EXPECTED_COUNTS:
            value = expected_summary[label, role]
            if int(row[role]) != value:
                raise ValueError('Class summary counts differ from the fixed split')
            expected_total += value
        if int(row['total']) != expected_total:
            raise ValueError('Invalid class summary total')
    provenance = read_json(folder / 'provenance.json')
    if provenance['dataset_id'] != DATASET_ID or provenance['membership_regenerated']:
        raise ValueError('Unexpected dataset provenance')
    if file_hash(folder / 'dataset.json') != provenance['source_manifest_sha256']:
        raise ValueError('The archived source manifest bytes have changed')
    return data, manifest


def check_image(path, item):
    path = Path(path)
    if file_hash(path) != item['sha256']:
        raise ValueError(f'Image file checksum mismatch: {path}')
    with Image.open(path) as image:
        image.load()
        if image.size != (int(item['width']), int(item['height'])):
            raise ValueError(f'Image dimensions differ: {path}')
        pixels = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    if pixels != item['decoded_sha256']:
        raise ValueError(f'Decoded RGB pixels differ: {path}')


def atomic_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite a different existing file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(source, output, spec, dry_run=False):
    data, manifest = load_spec(spec)
    source, output, spec = Path(source).resolve(), Path(output).resolve(), Path(spec).resolve()
    if not source.is_dir():
        raise ValueError(f'Source image directory does not exist: {source}')
    if source == output or spec == output:
        raise ValueError('Use a separate output directory')
    wanted = {row['sha256'] for row in manifest}
    matches = {}
    scanned = 0
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        if path.resolve().is_relative_to(output):
            continue
        scanned += 1
        checksum = file_hash(path)
        if checksum in wanted:
            matches.setdefault(checksum, path)
    missing = [row for row in manifest if row['sha256'] not in matches]
    if missing:
        details = [{'sample_id': r['sample_id'], 'relative_path': r['relative_path'],
                    'class_name': r['class_name']} for r in missing[:20]]
        raise ValueError(json.dumps({'missing_exact_images': len(missing),
            'examples': details, 'images_scanned': scanned,
            'input_requirement': 'Decoded image files matching the recorded checksums; native Phoenix files are not converted by this script'}, ensure_ascii=False))
    for item in manifest:
        check_image(matches[item['sha256']], item)
        target = safe_path(output, item['relative_path'])
        if target.exists():
            check_image(target, item)
    for name in SPEC_FILES:
        target = safe_path(output, name)
        if target.exists() and target.read_bytes() != (spec / name).read_bytes():
            raise ValueError(f'Existing output metadata differs: {target}')
    if not dry_run:
        for item in manifest:
            target = safe_path(output, item['relative_path'])
            if not target.exists():
                atomic_bytes(target, matches[item['sha256']].read_bytes())
                check_image(target, item)
        for name in SPEC_FILES:
            atomic_bytes(safe_path(output, name), (spec / name).read_bytes())
    return {'status': 'dry_run_passed' if dry_run else 'prepared',
            'dataset_id': DATASET_ID, 'samples': len(manifest),
            'counts': data['identity']['split_counts'], 'images_scanned': scanned,
            'output': str(output), 'split_regenerated': False}


def verify(folder, spec):
    data, manifest = load_spec(spec)
    folder = Path(folder).resolve()
    prepared = read_json(folder / 'dataset.json')
    if prepared != data:
        raise ValueError('Prepared dataset metadata differs from the frozen experiment')
    for item in manifest:
        check_image(safe_path(folder, item['relative_path']), item)
    return {'status': 'verified', 'dataset_id': DATASET_ID, 'samples': len(manifest),
            'counts': data['identity']['split_counts']}


def main():
    parser = argparse.ArgumentParser(description='Restore and verify the fixed MSTAR experiment subset')
    parser.add_argument('--metadata', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate')
    build = commands.add_parser('prepare')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--dry-run', action='store_true')
    check = commands.add_parser('verify')
    check.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            data, manifest = load_spec(args.metadata)
            result = {'status': 'metadata_valid', 'dataset_id': DATASET_ID,
                      'samples': len(manifest), 'counts': data['identity']['split_counts']}
        elif args.command == 'prepare':
            result = prepare(args.source, args.output, args.metadata, args.dry_run)
        else:
            result = verify(args.data, args.metadata)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(json.dumps({'status': 'error', 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
