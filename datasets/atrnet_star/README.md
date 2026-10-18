# ATRNet-STAR dataset

This directory contains the exact sample manifest and fixed partitions used by the ATRNet-STAR experiment. Image data and XML annotations are not redistributed.

## Official source

- [Official ATRNet-STAR project](https://github.com/waterdisappear/ATRNet-STAR)
- [Official dataset download and access terms](https://huggingface.co/datasets/waterdisappear/ATRNet-STAR)

Obtain the ground-range uint8 amplitude images and their XML annotations from the official SOC-40 release. Follow the access procedure and usage terms on the official pages, and cite the dataset authors. The download page requires an account and acceptance of its conditions.

## Files

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Validate metadata, restore the fixed subset from local official files, and verify images and annotations. |
| `dataset.json` | Frozen experiment identity, sample records, original source paths, prepared paths, labels, and checksums. |
| `manifest.csv` | Complete ordered sample list. |
| `train.csv` | Fixed training samples in their original order. |
| `val.csv` | Fixed validation samples in their original order. |
| `test.csv` | Fixed test samples in their original order. |
| `label_mapping.json` | Class names and numeric label mappings. |
| `class_counts.csv` | Sample counts by class and split. |
| `condition_counts.csv` | Retained-sample counts by imaging condition. |
| `group_assignment.csv` | Original geometry-group assignments used during subset construction. |
| `audit.json` | Original dataset-construction audit; it is not a model evaluation report. |
| `provenance.json` | Source archive identity, export changes, and metadata checksums. |
| `README.md` | Official links, file purposes, and preparation instructions. |

## Fixed subset

This is a custom ten-class subset of the official SOC-40 data, not an official SOC10 benchmark. It contains 4,800 images: 3,360 training images, 720 validation images, and 720 test images. Each class has 336 training, 72 validation, and 72 test images.

Training and validation samples come from the official training partition; test samples come from the official test partition. The archived memberships, class labels, geometry-group assignments, and sample order are preserved. The script does not select classes, randomly split samples, or replace missing records. Groups are geometry proxies and do not establish independent acquisition sequences or unseen vehicles.

`dataset.json` retains its original identity. Only the top-level `source_root` is changed to `.` for portability. Prepared images and XML files follow the relative `copied_image` and `copied_xml` paths recorded in each sample. The existing experiment loader can read the prepared directory in copied-image mode.

## Preparation and verification

Use Python 3.9 or later with NumPy and Pillow installed. Extract the official image and XML archives locally, then run from the repository root:

```bash
python -m pip install numpy Pillow
python datasets/atrnet_star/prepare_dataset.py validate
python datasets/atrnet_star/prepare_dataset.py prepare --source "/path/to/extracted_atrnet_star" --output "/path/to/prepared_atrnet_star" --dry-run
python datasets/atrnet_star/prepare_dataset.py prepare --source "/path/to/extracted_atrnet_star" --output "/path/to/prepared_atrnet_star"
python datasets/atrnet_star/prepare_dataset.py verify --data "/path/to/prepared_atrnet_star"
```

The script searches recursively and matches both images and XML files by their recorded SHA-256 hashes. It checks the original image dimensions, decoded pixel hashes, XML labels, and imaging metadata before copying. It preserves original bytes without resizing, re-encoding, or additional logarithmic conversion. Different existing output files and missing exact matches cause an error.

The source must contain the released single-frame, single-channel ground-range uint8 TIFF chips and their matching XML files. Complex slant-range data, full-scene raw data, and RGB auxiliary images are different products and are not converted by this script. Use the release corresponding to the recorded filenames and checksums; a changed release is not silently accepted.

The `verify` command checks all prepared images, including the test partition. It only verifies data integrity and does not train a model, calculate recognition scores, or use test performance to change the subset.
