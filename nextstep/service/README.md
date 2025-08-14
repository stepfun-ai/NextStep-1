# NextStep Service Module

The `nextstep/service` module provides data preview and visualization services to help you inspect and validate dataset contents.

---

## Overview

This module provides a Streamlit-based web interface for:

- ✅ Previewing individual tar file contents (images, videos, text, etc.)
- ✅ Previewing registered datasets (read from `pretrain_data.json`)
- ✅ Browsing dataset samples with forward/backward navigation
- ✅ Validating data format compliance

---

## Quick Start

### Launch Preview Service

```bash
streamlit run nextstep/service/_preview.py --server.port 8501
```

After launching, access `http://localhost:8501` in your browser.

### Feature Selection

The left sidebar provides two preview functions:

1. **preview_tar**: Preview individual tar files
   - Enter tar file path in the sidebar
   - Set number of display columns (1-10)
   - Automatically displays all samples in the tar file (up to 1000 samples)

2. **preview_dataset**: Preview registered datasets
   - Select dataset from dropdown list
   - Display dataset statistics (type, sample count, meta.json status)
   - Support forward/backward browsing or jump to specific index

---

## Module Structure

```
nextstep/service/
├── README.md              # This document
├── __init__.py            # Module initialization
├── _preview.py            # Streamlit main entry point
├── preview_tar.py         # Tar file preview functionality
├── preview_dataset.py     # Dataset preview functionality
└── utils.py               # Utility functions
```

### File Descriptions

| File | Description |
|------|-------------|
| `_preview.py` | Streamlit application main entry, provides feature selection interface |
| `preview_tar.py` | Implements preview functionality for individual tar files, supports images, videos, JSON, etc. |
| `preview_dataset.py` | Implements preview functionality for registered datasets, reads dataset information from `pretrain_data.json` |
| `utils.py` | Provides utility functions: session state management, image resizing, etc. |

---

## Use Cases

### Scenario 1: Inspect Newly Built Tar Files

After converting data to WebDataset format, use `preview_tar` to check:
- Whether tar file contents are correct
- Whether images and text match
- Whether data format meets requirements

### Scenario 2: Validate Dataset Registration

After registering datasets to `pretrain_data.json`, use `preview_dataset` to verify:
- Whether dataset paths are correct
- Whether `meta.json` files exist
- Whether sample counts match
- Whether data contents are normal

### Scenario 3: Debug Data Issues

When encountering data-related issues during training, use the preview service to:
- Inspect specific sample contents
- Verify data filtering conditions
- Confirm caption and image correspondences

---

## Notes

- **Performance**: `preview_tar` displays up to 1000 samples; large files may take longer to load
- **Caching**: The preview service caches processed data to improve repeated access speed
- **Paths**: Supports all path formats supported by `megfile` (local paths, S3, HTTP, etc.)
- **meta.json**: `preview_dataset` depends on `meta.json` files; if missing, it attempts to infer from the first tar file

---

## Related Documentation

- **Data Building**: `nextstep/data/build_wds.py` - Build WebDataset format data
- **Dataset Registration**: `configs/data/pretrain_data.json` - Dataset registration file
- **Data Indexing**: `nextstep/data/indexed_tar.py` - Tar file indexing implementation
