import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_spec(folder):
    folder = Path(folder).resolve()
    data = validate_manifest(read_json(folder / 'dataset.json'), DATASET_ID)
    samples = data['identity']['samples']
    if dict(Counter(row['split'] for row in samples)) != EXPECTED_COUNTS:
        raise ValueError('Unexpected split counts')
    for name, role in [('manifest.csv', None), ('train.csv', 'train'),
                       ('val.csv', 'validation'), ('test.csv', 'test')]:
        records = read_csv(folder / name)
        expected = [row for row in samples if role is None or row['split'] == role]
        if len(records) != len(expected):
            raise ValueError(f'CSV length mismatch: {name}')
        for index, (record, sample) in enumerate(zip(records, expected)):
            if set(record) != set(sample) or any(record[k] != str(sample[k]) for k in record):
                raise ValueError(f'CSV membership or order mismatch: {name}, row {index}')
    classes = data['identity']['classes']
    mapping = read_json(folder / 'label_mapping.json')
    if mapping != {'classes': classes,
                   'label_to_class': {str(i): name for i, name in enumerate(classes)},
                   'class_to_label': {name: i for i, name in enumerate(classes)}}:
        raise ValueError('Class mapping differs from the frozen experiment')
    provenance = read_json(folder / 'provenance.json')
    if provenance['dataset_id'] != DATASET_ID or provenance['memberships_regenerated']:
        raise ValueError('Unexpected dataset provenance')
    if set(provenance['metadata_sha256']) != set(ORIGINAL_FILES):
        raise ValueError('Incomplete metadata checksums')
    for name in ORIGINAL_FILES:
        if file_hash(folder / name) != provenance['metadata_sha256'][name]:
            raise ValueError(f'Metadata checksum mismatch: {name}')
    assignments = {row['group_id']: row['split'] for row in read_csv(folder / 'group_assignment.csv')}
    for sample in samples:
        if assignments.get(sample['group_id']) != sample['split']:
            raise ValueError('Geometry-group assignment differs from the frozen subset')
        if Path(sample['relative_path']).name != Path(sample['copied_image']).name:
            raise ValueError('Original and prepared image names differ')
        for key in ('sha256', 'xml_sha256', 'decoded_sha256'):
            value = sample[key]
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid sample checksum')
    return data


def check_pair(image, xml, row):
    if file_hash(image) != row['sha256']:
        raise ValueError(f'Image checksum mismatch: {image}')
    if file_hash(xml) != row['xml_sha256']:
        raise ValueError(f'XML checksum mismatch: {xml}')
    pixels = hashlib.sha256(read_sample_uint8(image).tobytes()).hexdigest()
    if pixels != row['decoded_sha256']:
        raise ValueError(f'Decoded pixels differ: {image}')
    check_xml(xml, Path(row['copied_image']), row)


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
    source, output, spec = Path(source).resolve(), Path(output).resolve(), Path(spec).resolve()
    data = load_spec(spec)
    samples = data['identity']['samples']
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    if source == output or spec == output:
        raise ValueError('Use a separate output directory')
    wanted = {row[key] for row in samples for key in ('sha256', 'xml_sha256')}
    matches = {}
    scanned = 0
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.tif', '.tiff', '.xml'}:
            continue
        if path.resolve().is_relative_to(output):
            continue
        scanned += 1
        checksum = file_hash(path)
        if checksum in wanted:
            matches.setdefault(checksum, path)
    missing_images = [row for row in samples if row['sha256'] not in matches]
    missing_xml = [row for row in samples if row['xml_sha256'] not in matches]
    if missing_images or missing_xml:
        examples = [{'image': row['relative_path'], 'xml': row['xml_path'], 'split': row['split']}
                    for row in (missing_images + missing_xml)[:20]]
        raise ValueError(json.dumps({'missing_images': len(missing_images), 'missing_xml': len(missing_xml),
            'examples': examples, 'files_scanned': scanned,
            'input_requirement': 'Exact ground-range uint8 TIFF chips and XML files from the official SOC-40 release'}))
    for row in samples:
        check_pair(matches[row['sha256']], matches[row['xml_sha256']], row)
        for path_key, hash_key in [('copied_image', 'sha256'), ('copied_xml', 'xml_sha256')]:
            target = safe_path(output, row[path_key])
            if target.exists() and file_hash(target) != row[hash_key]:
                raise ValueError(f'Existing output file differs: {target}')
    for name in SPEC_FILES:
        target = safe_path(output, name)
        if target.exists() and target.read_bytes() != (spec / name).read_bytes():
            raise ValueError(f'Existing output metadata differs: {target}')
    if not dry_run:
        for row in samples:
            for path_key, hash_key in [('copied_image', 'sha256'), ('copied_xml', 'xml_sha256')]:
                atomic_bytes(safe_path(output, row[path_key]), matches[row[hash_key]].read_bytes())
            check_pair(safe_path(output, row['copied_image']), safe_path(output, row['copied_xml']), row)
        for name in SPEC_FILES:
            atomic_bytes(safe_path(output, name), (spec / name).read_bytes())
    return {'status': 'dry_run_passed' if dry_run else 'prepared', 'dataset_id': DATASET_ID,
            'samples': len(samples), 'counts': EXPECTED_COUNTS, 'files_scanned': scanned,
            'output': str(output), 'split_regenerated': False}


def verify(folder, spec):
    folder, spec = Path(folder).resolve(), Path(spec).resolve()
    data = load_spec(spec)
    for name in SPEC_FILES:
        if (folder / name).read_bytes() != (spec / name).read_bytes():
            raise ValueError(f'Prepared metadata differs: {name}')
    for row in data['identity']['samples']:
        check_pair(safe_path(folder, row['copied_image']), safe_path(folder, row['copied_xml']), row)
    return {'status': 'verified', 'dataset_id': DATASET_ID,
            'samples': len(data['identity']['samples']), 'counts': EXPECTED_COUNTS}


def main():
    parser = argparse.ArgumentParser(description='Restore the exact fixed ATRNet-STAR experiment subset')
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
            data = load_spec(args.metadata)
            result = {'status': 'metadata_valid', 'dataset_id': DATASET_ID,
                      'samples': len(data['identity']['samples']), 'counts': EXPECTED_COUNTS}
        elif args.command == 'prepare':
            result = prepare(args.source, args.output, args.metadata, args.dry_run)
        else:
            result = verify(args.data, args.metadata)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ET.ParseError) as exc:
        print(json.dumps({'status': 'error', 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

DATASET_ID = "d1c7352bbb926c28b4e0665ca6b48c64c429d36ffdcd0b267bd8d4db717c0370"
EXPECTED_COUNTS = {"train": 3360, "validation": 720, "test": 720}
ROLES = ("train", "validation", "test")
ORIGINAL_FILES = ("dataset.json", "manifest.csv", "train.csv", "val.csv", "test.csv",
                  "label_mapping.json", "class_counts.csv", "condition_counts.csv",
                  "group_assignment.csv", "audit.json")
SPEC_FILES = ORIGINAL_FILES + ("provenance.json",)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_sample_uint8(path):
    with Image.open(Path(path)) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError(f'Multi-frame image is not supported: {path}')
        array = np.asarray(im)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError(f'Expected single-channel uint8, got {array.dtype}/{array.shape}: {path}')
        if array.shape != (128, 128):
            raise ValueError(f'Expected 128x128 chip, got {array.shape}: {path}')
        if array.max() == array.min():
            raise ValueError(f'Constant image: {path}')
        return np.array(array, copy=True)

def safe_path(root, relative):
    p = Path(relative)
    root = Path(root).resolve()
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe manifest path: ' + str(relative))
    out = (root / p).resolve()
    if root not in out.parents:
        raise ValueError('Manifest path escapes data root')
    return out

def geometry(row, sector):
    return [row['class'], row['target_id'], row['scene'], int(row['azimuth'] // sector)]

def validate_manifest(data, expected_dataset_id=None, exact_default=True):
    identity = data['identity']
    if digest(identity) != data['dataset_id']:
        raise ValueError('Frozen dataset checksum changed')
    if expected_dataset_id is not None and data['dataset_id'] != expected_dataset_id:
        raise ValueError('Dataset ID differs from the frozen audited dataset; no re-selection is allowed')
    classes = identity['classes']
    request = identity['request']
    rows = identity['samples']
    if len(classes) != 10 or len(set(classes)) != 10 or request['num_classes'] != 10:
        raise ValueError('This CNN requires exactly 10 distinct classes')
    quotas = request['per_class']
    if exact_default and (quotas != [336, 72, 72] or request['depressions'] != [15.0, 30.0] or request['sector_degrees'] != 30 or (request['guard_degrees'] != 2.5)):
        raise ValueError('Expected audited N4800 15/30-degree geometry protocol')
    if len(rows) != 10 * sum(quotas):
        raise ValueError('Manifest total quota differs')
    counts = Counter()
    seen = defaultdict(lambda: defaultdict(set))
    for row in rows:
        role = row['split']
        label = row['label']
        if role not in ROLES or not isinstance(label, int) or (not 0 <= label < 10) or (classes[label] != row['class']):
            raise ValueError('Invalid role/label')
        if row['source_split'] != ('test' if role == 'test' else 'train'):
            raise ValueError('Original source boundary violated')
        if row['depression'] not in request['depressions']:
            raise ValueError('Unexpected depression')
        offset = row['azimuth'] % request['sector_degrees']
        guard = request['guard_degrees']
        if guard and (not guard <= offset < request['sector_degrees'] - guard):
            raise ValueError('Angular guard violated')
        counts[role, label] += 1
        for key in ['relative_path', 'xml_path', 'copied_image', 'copied_xml']:
            path = Path(row[key])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe manifest path')
        for key in ['relative_path', 'xml_path', 'sha256', 'xml_sha256', 'decoded_sha256', 'group_id', 'copied_image', 'copied_xml']:
            seen[key][role].add(row[key])
        seen['geometry'][role].add(digest(geometry(row, request['sector_degrees'])))
    for role, quota in zip(ROLES, quotas):
        if any((counts[role, label] != quota for label in range(10))):
            raise ValueError('Class quota differs')
        for key in ['relative_path', 'sha256', 'decoded_sha256', 'copied_image', 'copied_xml']:
            if len(seen[key][role]) != 10 * quota:
                raise ValueError('Duplicate within split')
    for values in seen.values():
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            if values[a] & values[b]:
                raise ValueError('Cross-split hash/path/group overlap')
    return data

def check_xml(path, image, row):
    t = ET.parse(path).getroot()
    if t.findtext('filename') != image.name or t.findtext('object/type') != row['class']:
        raise ValueError('XML filename/label differs')
    for key, value in [('height', 128), ('width', 128), ('depth', 1)]:
        if int(t.findtext('image/' + key, '-1')) != value:
            raise ValueError('Invalid XML dimensions')
    if t.findtext('image/format') != 'ground_range_uint8':
        raise ValueError('Invalid XML format')
    for field, key in [('object/target_id', 'target_id'), ('scene/scene_name', 'scene')]:
        if (t.findtext(field) or '').strip() != row[key]:
            raise ValueError('XML metadata differs: ' + key)
    for field, key in [('sensor/band', 'band'), ('sensor/polarization', 'polarization')]:
        if (t.findtext(field) or '').strip().upper() != row[key]:
            raise ValueError('XML metadata differs: ' + key)
    for field, key in [('sensor/depression_angle', 'depression'), ('sensor/target_azimuth_angle', 'azimuth')]:
        value = float(re.sub('[°\\s]', '', t.findtext(field) or ''))
        if not math.isfinite(value):
            raise ValueError('Nonfinite XML angle')
        if key == 'azimuth':
            value %= 360
        if abs(value - row[key]) > 1e-06:
            raise ValueError('XML angle differs')


if __name__ == "__main__":
    raise SystemExit(main())
