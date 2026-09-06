# Document Scanner

AI powered document scanner built with Python and OpenCV.

## Features

- Image loading
- Preprocessing
- Edge detection
- Document detection
- Perspective correction
- Image enhancement
- PDF generation (Coming Soon)

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

Open preview windows after saving output files:

```bash
python -m app.main document.jpeg --show
```
