# MSTAR dataset

This directory contains the sample records and fixed splits used by the MSTAR experiment. Image data are not included.

## Official source

- [MSTAR Public Dataset — AFRL SDMS](https://www.sdms.afrl.af.mil/index.php?collection=mstar)
- [SDMS download access and account registration](https://www.sdms.afrl.af.mil/index.php?collection=registration)

Obtain the dataset through the official portal and follow its access and usage terms. 

## Files

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Validate metadata, restore the fixed directory structure from matching image files, and verify the resulting dataset. |
| `dataset.json` | Original experiment manifest, with relative image paths, labels, split membership, and file and pixel checksums. |
| `sample_manifest.csv` | Readable sample list with identifiers, filenames, classes, image dimensions, and checksums. |
| `splits.csv` | Fixed training, validation, and test membership and sample order. |
| `class_mapping.json` | Mapping from numeric labels to class names. |
| `split_summary.csv` | Sample counts by class and split. |
| `provenance.json` | Source archive and manifest identities and the decoded-image checksum definition. |
| `README.md` | Official links, file purposes, and dataset preparation instructions. |

## Fixed subset

The subset contains 5,172 images from 10 classes: 3,617 training images, 776 validation images, and 779 test images. See `split_summary.csv` for class counts and `class_mapping.json` for class names.

The original memberships and within-split order are retained. No random splitting, deduplication, or resampling is performed. In `splits.csv`, `split_index` is the zero-based position within its split. The `validation` split corresponds to the original split file's `val` key.

Prepared images are stored at `Images/<label>/<filename>` relative to the prepared dataset directory. File checksums are SHA-256 hashes of the original file bytes. Decoded checksums are SHA-256 hashes of `PIL.Image.convert("RGB").tobytes()` at the original dimensions.

## Preparation and verification

Use Python 3.9 or later with Pillow installed. Run these commands from the repository root:

```bash
python -m pip install Pillow
python datasets/mstar/prepare_dataset.py validate
python datasets/mstar/prepare_dataset.py prepare --source "/path/to/decoded_mstar_images" --output "/path/to/prepared_mstar" --dry-run
python datasets/mstar/prepare_dataset.py prepare --source "/path/to/decoded_mstar_images" --output "/path/to/prepared_mstar"
python datasets/mstar/prepare_dataset.py verify --data "/path/to/prepared_mstar"
```

The script searches the source directory recursively and matches images by file checksum, because filenames can repeat across classes. It checks every selected image's bytes, dimensions, and decoded pixels before copying. Existing files with different contents are rejected. Missing or changed images produce an error; the script never substitutes samples or generates a new split.

The input must contain the exact prepared JPEG files recorded by the experiment. The archived experiment does not provide a native Phoenix-to-JPEG conversion pipeline, and this script does not implement that conversion. Downloading the official raw data alone therefore does not establish an exact reconstruction of the recorded JPEG files. Re-encoding or resizing images can change their checksums. Image transformations used during model training belong in the training code and are not applied here.
