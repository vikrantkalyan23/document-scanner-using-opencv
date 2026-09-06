# Document Scanner

AI powered document scanner built with Python and OpenCV.

## Features

- Image loading
- Preprocessing
- Edge detection
- Document detection
- Perspective correction
- Image enhancement
- Noise reduction and shadow cleanup
- Multiple scan modes
- Batch processing
- PDF generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run:

```bash
python -m app.main
```

Scan a different image from the `input` directory:

```bash
python -m app.main receipt.jpg
```

Choose an enhancement mode:

```bash
python -m app.main receipt.jpg --mode bw
python -m app.main receipt.jpg --mode color
python -m app.main receipt.jpg --mode gray
python -m app.main receipt.jpg --mode soft
python -m app.main receipt.jpg --mode ocr
```

Scan every supported image in the `input` directory:

```bash
python -m app.main --batch
```

Save enhanced scans as a PDF:

```bash
python -m app.main document.jpeg --pdf
python -m app.main --batch --pdf
```

Open preview windows after saving output files:

```bash
python -m app.main document.jpeg --show
```
