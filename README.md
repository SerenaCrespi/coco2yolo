# coco2yolo

This repository contains a utility to convert page-level letter annotations
into line-level datasets for YOLO segmentation.

The script was initially developed for annotations produced with our custom
letter annotation workflow. In this workflow, handwritten letters are
annotated as polygons on full manuscript pages and exported in COCO format.

The annotations are therefore structured at page level:

- one full-page image;
- COCO polygon annotations for individual letters;
- one ALTO XML file describing the text lines of the same page.

The script uses the ALTO line information to redefine the dataset at
line level.

## What does the script do?

Starting from:

```text
input/
├── images/
│   └── page1.png
├── alto/
│   └── page1.xml
└── coco/
    └── instances_all.json
```

the script:

1. reads the text lines from the ALTO file;
2. assigns each annotated letter to the most appropriate text line;
3. crops the full page into individual line images;
4. recalculates the letter polygon coordinates for each crop;
5. creates YOLO segmentation labels;
6. creates a new line-level COCO dataset;
7. generates debug images to check that the polygons were transferred correctly.

## Why is this necessary?

The original annotations are created on complete manuscript pages.

For training letter-level instance segmentation models, however, it can be
more useful to work with cropped text lines, where individual letters occupy
a larger portion of the image.

The script therefore transforms:

```text
full page + page-level letter polygons
```

into:

```text
cropped text lines + line-level letter polygons
```

without manually annotating the lines again.

## How are the annotations transferred?

The ALTO file provides the position of each text line.

Each letter annotation is first assigned to a line using its position
relative to the ALTO baselines.

Once the line is cropped, the polygon coordinates are recalculated relative
to the top-left corner of the new crop.

For example, if a polygon point on the original page is:

```text
(1250, 1280)
```

and the line crop starts at:

```text
(450, 1100)
```

the new point becomes:

```text
(800, 180)
```

because:

```text
x_crop = x_page - crop_left
y_crop = y_page - crop_top
```

The polygon shape itself is not resized or modified.

## Output

The script creates:

```text
output/
├── images/
├── labels/
├── coco/
├── transcriptions/
├── review/
├── debug/
│   ├── pages/
│   └── lines/
└── classes.txt
```

### `images/`

Cropped text-line images.

### `labels/`

YOLO segmentation labels for each cropped line.

### `coco/`

A new COCO dataset using the cropped line images and recalculated polygon
coordinates.

### `transcriptions/`

ALTO transcriptions associated with the cropped lines.

### `review/`

Information about letter-to-line assignments that may require manual review.

### `debug/pages/`

Full-page visualizations showing the ALTO lines and the assignment of
letter annotations to them.

### `debug/lines/`

Cropped lines with the transferred letter polygons drawn on top.

These files make it possible to visually verify that the annotations were
correctly reassigned.

## Installation

Only Pillow is required:

```bash
pip install Pillow
```

## Usage

Place the images, ALTO files and COCO annotations inside the `input` folder:

```text
input/
├── images/
├── alto/
└── coco/
```

Then run:

```bash
python3 crop_alto_lines.py
```

The converted dataset will be created automatically inside:

```text
output/
```

No manifest file is required.

## Generalisation

The script was developed for our specific annotation workflow, but the
conversion logic is not tied to the annotator itself.

It can be adapted to other datasets as long as they provide:

- page images;
- polygon annotations in COCO format;
- line coordinates or baselines that can be used to crop the page.

ALTO is currently used as the source of line geometry, but the same approach
could be generalised to other line segmentation formats.
