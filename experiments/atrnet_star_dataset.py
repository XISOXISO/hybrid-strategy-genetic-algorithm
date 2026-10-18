from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from collections import Counter, defaultdict
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from atrnet_star_common import digest, file_hash, read_json
from atrnet_star_atr_image import read_sample_uint8
ROLES = ('train', 'validation', 'test')

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

def load_prepared(folder, root_override=None, verify_train_val=False, progress=None, expected_dataset_id=None, image_source='copied', exact_default=True):
    folder = Path(folder).resolve()
    data = validate_manifest(read_json(folder / 'dataset.json'), expected_dataset_id, exact_default)
    identity = data['identity']
    rows = identity['samples']
    for name, role in [('manifest.csv', None), ('train.csv', 'train'), ('val.csv', 'validation'), ('test.csv', 'test')]:
        with (folder / name).open(encoding='utf-8-sig', newline='') as f:
            csvrows = list(csv.DictReader(f))
        expected = [r for r in rows if role is None or r['split'] == role]
        if len(csvrows) != len(expected):
            raise ValueError('CSV length differs: ' + name)
        for a, b in zip(csvrows, expected):
            if set(a) != set(b) or any((a[k] != str(b[k]) for k in a)):
                raise ValueError('CSV differs: ' + name)
    if image_source not in {'copied', 'original'}:
        raise ValueError('Invalid image source')
    if image_source == 'copied' and identity['request']['copy_mode'] != 'copy':
        raise ValueError('Frozen dataset has no image copies; use --image-source original with a new output')
    root = folder if image_source == 'copied' else Path(root_override or data['source_root']).resolve()
    samples = {'train': [], 'validation': []}
    selected = [r for r in rows if r['split'] in samples]
    for i, row in enumerate(selected, 1):
        ikey, xkey = ('copied_image', 'copied_xml') if image_source == 'copied' else ('relative_path', 'xml_path')
        image = safe_path(root, row[ikey])
        xml = safe_path(root, row[xkey])
        if not image.is_file() or not xml.is_file():
            raise FileNotFoundError(str(image) + ' / ' + str(xml))
        if verify_train_val:
            if file_hash(image) != row['sha256'] or file_hash(xml) != row['xml_sha256']:
                raise ValueError('Frozen training/validation image or XML changed: ' + str(image))
            check_xml(xml, image, row)
            if hashlib.sha256(read_sample_uint8(image).tobytes()).hexdigest() != row['decoded_sha256']:
                raise ValueError('Frozen decoded pixels changed')
        samples[row['split']].append((str(image), row['label']))
        if progress and (i % 25 == 0 or i == len(selected)):
            progress(i / len(selected), f'Verified train/validation {i}/{len(selected)}; test untouched')
    return (data, samples)
