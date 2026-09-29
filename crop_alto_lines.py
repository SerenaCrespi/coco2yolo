#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert page-level COCO letter annotations into line-level crops
using ALTO TextLines.

Expected input structure:

input/
├── images/
│   ├── page1.png
│   └── ...
├── alto/
│   ├── page1.xml
│   └── ...
└── coco/
    └── instances_all.json

No manifest is required.

Workflow
--------

1. Match image / ALTO / COCO automatically.
2. Read ALTO TextLines and baselines.
3. Assign each COCO annotation to the nearest ALTO baseline.
4. Compute a safe crop for each line:
   - start from the ALTO TextLine bounding box;
   - add proportional vertical padding;
   - add horizontal padding;
   - expand further to include assigned letter polygons;
   - limit excessive expansion caused by possible assignment errors.
5. Crop each line.
6. Convert page polygon coordinates into crop coordinates.
7. Create:
   - cropped line images;
   - YOLO segmentation labels;
   - line-level COCO JSON;
   - transcription CSV;
   - assignment review CSV;
   - page-level debug images;
   - line-level debug images.

Output structure:

output/
├── images/
├── labels/
├── coco/
│   └── instances_lines.json
├── transcriptions/
│   └── transcriptions.csv
├── review/
│   └── assignment_review.csv
├── debug/
│   ├── pages/
│   └── lines/
└── classes.txt

Dependencies:
    Pillow

Usage:
    python3 crop_alto_lines.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ALTO namespace
# ============================================================

ALTO_NS = {
    "a": "http://www.loc.gov/standards/alto/ns-v4#"
}


# ============================================================
# Data structures
# ============================================================

@dataclass
class AltoLine:
    """Representation of one ALTO TextLine."""

    line_id: str
    x: int
    y: int
    w: int
    h: int
    text: str
    baseline: list[tuple[float, float]]


# ============================================================
# General helpers
# ============================================================

def safe_name(value: str) -> str:
    """Convert a string into a safe filename."""

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value.strip()
    )

    return value.strip("._") or "line"


def get_font():
    """Load Pillow's default font."""

    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ============================================================
# ALTO reading
# ============================================================

def read_alto(
    path: Path
) -> tuple[list[AltoLine], tuple[int, int]]:
    """
    Read all ALTO TextLines.

    Returns:
        lines
        page_size = (width, height)
    """

    root = ET.parse(path).getroot()

    page = root.find(
        ".//a:Page",
        ALTO_NS
    )

    if page is None:
        raise ValueError(
            f"No ALTO Page element found in {path}"
        )

    page_size = (
        int(float(page.get("WIDTH", 0))),
        int(float(page.get("HEIGHT", 0))),
    )

    lines: list[AltoLine] = []

    for index, node in enumerate(
        root.findall(
            ".//a:TextLine",
            ALTO_NS
        )
    ):

        # ----------------------------------------------------
        # Read transcription
        # ----------------------------------------------------

        strings = node.findall(
            "a:String",
            ALTO_NS
        )

        text = " ".join(
            s.get("CONTENT", "")
            for s in strings
            if s.get("CONTENT")
        ).strip()

        # ----------------------------------------------------
        # Read baseline
        # ----------------------------------------------------

        baseline_raw = node.get(
            "BASELINE",
            ""
        ).strip()

        baseline: list[
            tuple[float, float]
        ] = []

        if baseline_raw:

            try:

                values = [
                    float(v)
                    for v in baseline_raw.split()
                ]

                if len(values) % 2 != 0:
                    raise ValueError(
                        "BASELINE must contain x/y pairs"
                    )

                baseline = list(
                    zip(
                        values[0::2],
                        values[1::2]
                    )
                )

            except ValueError as exc:

                print(
                    f"WARNING: invalid BASELINE "
                    f"in {path.name}: {exc}",
                    file=sys.stderr
                )

        lines.append(
            AltoLine(
                line_id=node.get(
                    "ID",
                    f"line_{index:04d}"
                ),
                x=int(float(node.get("HPOS", 0))),
                y=int(float(node.get("VPOS", 0))),
                w=int(float(node.get("WIDTH", 0))),
                h=int(float(node.get("HEIGHT", 0))),
                text=text,
                baseline=baseline,
            )
        )

    return lines, page_size


def alto_source_filename(
    path: Path
) -> str:
    """Return the image filename declared inside ALTO."""

    root = ET.parse(path).getroot()

    node = root.find(
        ".//a:sourceImageInformation/a:fileName",
        ALTO_NS
    )

    if node is None:
        return ""

    return (
        node.text or ""
    ).strip()


# ============================================================
# Automatic input discovery
# ============================================================

def find_alto_file(
    alto_dir: Path,
    image_path: Path
) -> Path | None:
    """Find the ALTO XML corresponding to an image."""

    candidates = [
        alto_dir / f"{image_path.stem}.xml",
        alto_dir / f"{image_path.name}.xml",
    ]

    for candidate in candidates:

        if candidate.is_file():
            return candidate

    return None


def load_coco_files(
    coco_dir: Path
) -> list[tuple[Path, dict]]:
    """Load all valid COCO JSON files."""

    paths = sorted(
        coco_dir.glob("*.json")
    )

    if not paths:

        raise FileNotFoundError(
            f"No COCO JSON files found in {coco_dir}"
        )

    loaded = []

    for path in paths:

        try:

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:

            print(
                f"WARNING: cannot read {path}: {exc}",
                file=sys.stderr
            )
            continue

        if (
            "images" not in data
            or
            "annotations" not in data
        ):

            print(
                f"WARNING: {path.name} does not look like a COCO dataset.",
                file=sys.stderr
            )
            continue

        loaded.append(
            (
                path,
                data
            )
        )

    if not loaded:

        raise ValueError(
            "No valid COCO dataset could be loaded."
        )

    return loaded


def find_coco_image(
    image_path: Path,
    coco_files: list[tuple[Path, dict]]
) -> tuple[Path, dict, dict] | None:
    """Find the COCO image record corresponding to the page image."""

    matches = []

    for coco_path, coco_data in coco_files:

        local_matches = []

        for image_entry in coco_data.get(
            "images",
            []
        ):

            file_name = image_entry.get(
                "file_name",
                ""
            )

            coco_name = Path(
                PurePosixPath(
                    file_name
                ).name
            ).name

            if coco_name == image_path.name:

                local_matches.append(
                    image_entry
                )

        if len(local_matches) > 1:

            raise ValueError(
                f"Multiple COCO entries found for "
                f"{image_path.name} in {coco_path}"
            )

        if local_matches:

            matches.append(
                (
                    coco_path,
                    coco_data,
                    local_matches[0]
                )
            )

    if not matches:
        return None

    if len(matches) > 1:

        raise ValueError(
            f"{image_path.name} appears in multiple COCO files."
        )

    return matches[0]


def discover_pages(
    input_dir: Path
) -> list[dict]:
    """
    Automatically match:

        image
        ALTO
        COCO
    """

    images_dir = input_dir / "images"
    alto_dir = input_dir / "alto"
    coco_dir = input_dir / "coco"

    for directory in (
        images_dir,
        alto_dir,
        coco_dir
    ):

        if not directory.is_dir():

            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    images = sorted(
        p
        for p in images_dir.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower() in extensions
        )
    )

    if not images:

        raise FileNotFoundError(
            f"No images found in {images_dir}"
        )

    coco_files = load_coco_files(
        coco_dir
    )

    pages = []

    for image_path in images:

        alto_path = find_alto_file(
            alto_dir,
            image_path
        )

        if alto_path is None:

            print(
                f"WARNING: no ALTO for {image_path.name}",
                file=sys.stderr
            )
            continue

        coco_match = find_coco_image(
            image_path,
            coco_files
        )

        if coco_match is None:

            print(
                f"WARNING: {image_path.name} not found in COCO.",
                file=sys.stderr
            )
            continue

        (
            coco_path,
            coco_data,
            coco_image
        ) = coco_match

        pages.append(
            {
                "page_id":
                    image_path.stem,

                "folder":
                    safe_name(
                        image_path.stem
                    ),

                "image":
                    image_path,

                "alto":
                    alto_path,

                "coco":
                    coco_path,

                "coco_data":
                    coco_data,

                "coco_image":
                    coco_image,
            }
        )

    if not pages:

        raise ValueError(
            "No complete image / ALTO / COCO association found."
        )

    return pages


# ============================================================
# COCO polygon helpers
# ============================================================

def segmentation_polygons(
    annotation: dict
) -> list[list[float]]:
    """Extract polygon segmentations from a COCO annotation."""

    segmentation = annotation.get(
        "segmentation",
        []
    )

    if not isinstance(
        segmentation,
        list
    ):
        return []

    # Handle a single flat polygon if necessary.
    if (
        segmentation
        and
        all(
            isinstance(
                value,
                (int, float)
            )
            for value in segmentation
        )
    ):

        segmentation = [
            segmentation
        ]

    polygons = []

    for polygon in segmentation:

        if (
            isinstance(
                polygon,
                list
            )
            and
            len(polygon) >= 6
            and
            len(polygon) % 2 == 0
        ):

            polygons.append(
                [
                    float(v)
                    for v in polygon
                ]
            )

    return polygons


def annotation_bbox(
    annotation: dict
) -> tuple[
    float,
    float,
    float,
    float
]:
    """Return annotation bbox as x, y, width, height."""

    bbox = annotation.get(
        "bbox"
    )

    if (
        isinstance(
            bbox,
            list
        )
        and
        len(bbox) >= 4
    ):

        return tuple(
            float(v)
            for v in bbox[:4]
        )

    polygons = segmentation_polygons(
        annotation
    )

    points = [
        (
            polygon[i],
            polygon[i + 1]
        )
        for polygon in polygons
        for i in range(
            0,
            len(polygon),
            2
        )
    ]

    if not points:

        raise ValueError(
            f"Annotation {annotation.get('id')} "
            "has no valid bbox or polygon."
        )

    xs = [
        p[0]
        for p in points
    ]

    ys = [
        p[1]
        for p in points
    ]

    x1 = min(xs)
    x2 = max(xs)

    y1 = min(ys)
    y2 = max(ys)

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


# ============================================================
# Baseline geometry
# ============================================================

def baseline_y_at_x(
    line: AltoLine,
    x: float
) -> float:
    """
    Estimate baseline Y at coordinate X.

    If no ALTO baseline exists, use the vertical centre
    of the TextLine bounding box.
    """

    if not line.baseline:

        return (
            line.y
            +
            line.h / 2.0
        )

    points = sorted(
        line.baseline,
        key=lambda p: p[0]
    )

    if len(points) == 1:
        return points[0][1]

    if x <= points[0][0]:
        return points[0][1]

    if x >= points[-1][0]:
        return points[-1][1]

    for (
        (x1, y1),
        (x2, y2)
    ) in zip(
        points,
        points[1:]
    ):

        if x1 <= x <= x2:

            if x2 == x1:
                return (
                    y1 + y2
                ) / 2.0

            ratio = (
                x - x1
            ) / (
                x2 - x1
            )

            return (
                y1
                +
                ratio
                *
                (
                    y2 - y1
                )
            )

    return (
        line.y
        +
        line.h / 2.0
    )


def assign_annotation_to_line(
    annotation: dict,
    lines: list[AltoLine]
) -> tuple[int, float, float, float]:
    """
    Assign one COCO annotation to the closest ALTO baseline.

    The centre of the annotation bounding box is compared
    with every baseline.

    Returns:
        line_index
        baseline_distance
        centre_x
        centre_y
    """

    x, y, w, h = annotation_bbox(
        annotation
    )

    centre_x = (
        x + w / 2.0
    )

    centre_y = (
        y + h / 2.0
    )

    distances = []

    for line in lines:

        baseline_y = baseline_y_at_x(
            line,
            centre_x
        )

        distance = abs(
            centre_y
            -
            baseline_y
        )

        distances.append(
            distance
        )

    line_index = min(
        range(
            len(lines)
        ),
        key=distances.__getitem__
    )

    return (
        line_index,
        distances[line_index],
        centre_x,
        centre_y
    )


# ============================================================
# Crop geometry
# ============================================================

def compute_line_crop_bounds(
    line: AltoLine,
    assigned_annotations: list,
    page_width: int,
    page_height: int,
    horizontal_padding: int,
    vertical_padding_ratio: float,
    polygon_padding: int,
    max_expand_ratio: float,
    crop_expand_distance: float,
) -> tuple[int, int, int, int]:
    """
    Compute a safe crop rectangle for an ALTO line.

    Strategy:

    1. Start with the ALTO TextLine bounding box.
    2. Add horizontal padding.
    3. Add vertical padding proportional to line height.
    4. Look at COCO polygons assigned to the line.
    5. Expand the crop to include those polygons.
    6. Ignore suspiciously distant annotations for crop expansion.
    7. Limit the maximum expansion around the ALTO box.

    This prevents ascenders, descenders and flourishes from being cut.
    """

    vertical_padding = max(
        5,
        int(
            round(
                line.h
                *
                vertical_padding_ratio
            )
        )
    )

    # --------------------------------------------------------
    # Initial crop from ALTO geometry
    # --------------------------------------------------------

    base_left = (
        line.x
        -
        horizontal_padding
    )

    base_right = (
        line.x
        +
        line.w
        +
        horizontal_padding
    )

    base_top = (
        line.y
        -
        vertical_padding
    )

    base_bottom = (
        line.y
        +
        line.h
        +
        vertical_padding
    )

    left = base_left
    right = base_right
    top = base_top
    bottom = base_bottom

    # --------------------------------------------------------
    # Maximum allowed expansion
    #
    # This protects against a wrongly assigned annotation.
    # --------------------------------------------------------

    max_extra = max(
        vertical_padding,
        int(
            round(
                line.h
                *
                max_expand_ratio
            )
        )
    )

    allowed_left = (
        base_left
        -
        max_extra
    )

    allowed_right = (
        base_right
        +
        max_extra
    )

    allowed_top = (
        base_top
        -
        max_extra
    )

    allowed_bottom = (
        base_bottom
        +
        max_extra
    )

    # --------------------------------------------------------
    # Expand crop using trustworthy assigned polygons
    # --------------------------------------------------------

    for (
        annotation,
        distance,
        centre_x,
        centre_y
    ) in assigned_annotations:

        # Do not let a suspicious assignment change crop geometry.
        if distance > crop_expand_distance:
            continue

        polygons = segmentation_polygons(
            annotation
        )

        for polygon in polygons:

            xs = polygon[0::2]
            ys = polygon[1::2]

            if not xs or not ys:
                continue

            polygon_left = (
                min(xs)
                -
                polygon_padding
            )

            polygon_right = (
                max(xs)
                +
                polygon_padding
            )

            polygon_top = (
                min(ys)
                -
                polygon_padding
            )

            polygon_bottom = (
                max(ys)
                +
                polygon_padding
            )

            left = min(
                left,
                polygon_left
            )

            right = max(
                right,
                polygon_right
            )

            top = min(
                top,
                polygon_top
            )

            bottom = max(
                bottom,
                polygon_bottom
            )

    # --------------------------------------------------------
    # Apply maximum expansion limits
    # --------------------------------------------------------

    left = max(
        left,
        allowed_left
    )

    right = min(
        right,
        allowed_right
    )

    top = max(
        top,
        allowed_top
    )

    bottom = min(
        bottom,
        allowed_bottom
    )

    # --------------------------------------------------------
    # Clip to page boundaries
    # --------------------------------------------------------

    left = max(
        0,
        int(
            math.floor(
                left
            )
        )
    )

    top = max(
        0,
        int(
            math.floor(
                top
            )
        )
    )

    right = min(
        page_width,
        int(
            math.ceil(
                right
            )
        )
    )

    bottom = min(
        page_height,
        int(
            math.ceil(
                bottom
            )
        )
    )

    return (
        left,
        top,
        right,
        bottom
    )


# ============================================================
# Polygon clipping
# ============================================================

def clip_polygon_to_rect(
    polygon: list[float],
    left: float,
    top: float,
    right: float,
    bottom: float
) -> list[float]:
    """
    Clip a polygon to a rectangle using the
    Sutherland-Hodgman algorithm.
    """

    points = [
        (
            polygon[i],
            polygon[i + 1]
        )
        for i in range(
            0,
            len(polygon),
            2
        )
    ]

    if len(points) < 3:
        return []

    def clip_edge(
        points_in,
        inside,
        intersection
    ):

        if not points_in:
            return []

        output = []

        previous = points_in[-1]
        previous_inside = inside(
            previous
        )

        for current in points_in:

            current_inside = inside(
                current
            )

            if current_inside:

                if not previous_inside:

                    output.append(
                        intersection(
                            previous,
                            current
                        )
                    )

                output.append(
                    current
                )

            elif previous_inside:

                output.append(
                    intersection(
                        previous,
                        current
                    )
                )

            previous = current
            previous_inside = current_inside

        return output

    def intersect_vertical(
        p1,
        p2,
        x_value
    ):

        x1, y1 = p1
        x2, y2 = p2

        if x2 == x1:
            return (
                x_value,
                y1
            )

        ratio = (
            x_value - x1
        ) / (
            x2 - x1
        )

        return (
            x_value,
            y1
            +
            ratio
            *
            (
                y2 - y1
            )
        )

    def intersect_horizontal(
        p1,
        p2,
        y_value
    ):

        x1, y1 = p1
        x2, y2 = p2

        if y2 == y1:
            return (
                x1,
                y_value
            )

        ratio = (
            y_value - y1
        ) / (
            y2 - y1
        )

        return (
            x1
            +
            ratio
            *
            (
                x2 - x1
            ),
            y_value
        )

    points = clip_edge(
        points,
        lambda p:
            p[0] >= left,
        lambda a, b:
            intersect_vertical(
                a,
                b,
                left
            )
    )

    points = clip_edge(
        points,
        lambda p:
            p[0] <= right,
        lambda a, b:
            intersect_vertical(
                a,
                b,
                right
            )
    )

    points = clip_edge(
        points,
        lambda p:
            p[1] >= top,
        lambda a, b:
            intersect_horizontal(
                a,
                b,
                top
            )
    )

    points = clip_edge(
        points,
        lambda p:
            p[1] <= bottom,
        lambda a, b:
            intersect_horizontal(
                a,
                b,
                bottom
            )
    )

    if len(points) < 3:
        return []

    flattened = []

    for x, y in points:

        flattened.extend(
            [
                x,
                y
            ]
        )

    return flattened


# ============================================================
# Coordinate reconstruction
# ============================================================

def translate_polygon_to_crop(
    polygon: list[float],
    crop_left: float,
    crop_top: float
) -> list[float]:
    """
    Convert PAGE coordinates into LINE CROP coordinates.

    If the crop starts at:

        crop_left = 1000
        crop_top  = 2000

    then a page point:

        (1200, 2075)

    becomes:

        (200, 75)

    because:

        x_crop = x_page - crop_left
        y_crop = y_page - crop_top
    """

    result = []

    for i in range(
        0,
        len(polygon),
        2
    ):

        x_page = polygon[i]
        y_page = polygon[i + 1]

        x_crop = (
            x_page
            -
            crop_left
        )

        y_crop = (
            y_page
            -
            crop_top
        )

        result.extend(
            [
                x_crop,
                y_crop
            ]
        )

    return result


def normalize_polygon_for_yolo(
    polygon: list[float],
    crop_width: int,
    crop_height: int
) -> list[float]:
    """
    Convert crop pixel coordinates into YOLO normalized coordinates.

        x_yolo = x_crop / crop_width
        y_yolo = y_crop / crop_height

    YOLO values are therefore between 0 and 1.
    """

    result = []

    for i in range(
        0,
        len(polygon),
        2
    ):

        x = (
            polygon[i]
            /
            crop_width
        )

        y = (
            polygon[i + 1]
            /
            crop_height
        )

        x = min(
            max(
                x,
                0.0
            ),
            1.0
        )

        y = min(
            max(
                y,
                0.0
            ),
            1.0
        )

        result.extend(
            [
                x,
                y
            ]
        )

    return result


def polygon_area(
    polygon: list[float]
) -> float:
    """Calculate polygon area."""

    points = [
        (
            polygon[i],
            polygon[i + 1]
        )
        for i in range(
            0,
            len(polygon),
            2
        )
    ]

    if len(points) < 3:
        return 0.0

    area = 0.0

    for (
        (x1, y1),
        (x2, y2)
    ) in zip(
        points,
        points[1:] + points[:1]
    ):

        area += (
            x1 * y2
            -
            x2 * y1
        )

    return abs(
        area
    ) / 2.0


# ============================================================
# Debug drawing
# ============================================================

def draw_line_debug(
    crop: Image.Image,
    debug_polygons: list[dict],
    category_by_id: dict,
    output_path: Path
) -> None:
    """
    Draw transferred polygons over the final line crop.
    """

    image = crop.copy().convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )

    font = get_font()

    for item in debug_polygons:

        polygon = item[
            "polygon"
        ]

        category_id = item[
            "category_id"
        ]

        annotation_id = item[
            "annotation_id"
        ]

        points = [
            (
                polygon[i],
                polygon[i + 1]
            )
            for i in range(
                0,
                len(polygon),
                2
            )
        ]

        if len(points) < 3:
            continue

        draw.line(
            points
            +
            [
                points[0]
            ],
            fill="red",
            width=2
        )

        category_name = (
            category_by_id.get(
                category_id,
                {}
            ).get(
                "name",
                str(category_id)
            )
        )

        label = (
            f"{category_name} "
            f"#{annotation_id}"
        )

        label_x = min(
            p[0]
            for p in points
        )

        label_y = min(
            p[1]
            for p in points
        )

        draw.text(
            (
                label_x,
                max(
                    0,
                    label_y - 12
                )
            ),
            label,
            fill="red",
            font=font
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        output_path
    )


def draw_page_debug(
    page_image: Image.Image,
    lines: list[AltoLine],
    assignments: dict,
    category_by_id: dict,
    output_path: Path
) -> None:
    """
    Draw page-level assignment debug information.

    Shows:
        blue  = ALTO TextLine box
        green = ALTO baseline
        red   = annotation centre
        orange = distance from centre to assigned baseline
    """

    image = page_image.copy().convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )

    font = get_font()

    for line_index, line in enumerate(
        lines
    ):

        draw.rectangle(
            (
                line.x,
                line.y,
                line.x + line.w,
                line.y + line.h
            ),
            outline="blue",
            width=2
        )

        if len(
            line.baseline
        ) >= 2:

            draw.line(
                line.baseline,
                fill="green",
                width=2
            )

        draw.text(
            (
                line.x,
                max(
                    0,
                    line.y - 12
                )
            ),
            f"L{line_index}",
            fill="blue",
            font=font
        )

    for line_index, assigned in assignments.items():

        for (
            annotation,
            distance,
            centre_x,
            centre_y
        ) in assigned:

            category_id = int(
                annotation[
                    "category_id"
                ]
            )

            category_name = (
                category_by_id.get(
                    category_id,
                    {}
                ).get(
                    "name",
                    str(category_id)
                )
            )

            radius = 4

            draw.ellipse(
                (
                    centre_x - radius,
                    centre_y - radius,
                    centre_x + radius,
                    centre_y + radius
                ),
                fill="red"
            )

            baseline_y = baseline_y_at_x(
                lines[
                    line_index
                ],
                centre_x
            )

            draw.line(
                (
                    centre_x,
                    centre_y,
                    centre_x,
                    baseline_y
                ),
                fill="orange",
                width=1
            )

            draw.text(
                (
                    centre_x + 4,
                    centre_y
                ),
                f"{category_name}→L{line_index}",
                fill="red",
                font=font
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        output_path
    )


# ============================================================
# Dataset processing
# ============================================================

def process_dataset(
    pages: list[dict],
    output_dir: Path,
    horizontal_padding: int,
    vertical_padding_ratio: float,
    polygon_padding: int,
    max_expand_ratio: float,
    crop_expand_distance: float,
    review_distance: float
) -> None:
    """Process all pages."""

    images_out = (
        output_dir
        /
        "images"
    )

    labels_out = (
        output_dir
        /
        "labels"
    )

    coco_out = (
        output_dir
        /
        "coco"
    )

    transcription_out = (
        output_dir
        /
        "transcriptions"
    )

    review_out = (
        output_dir
        /
        "review"
    )

    debug_lines_out = (
        output_dir
        /
        "debug"
        /
        "lines"
    )

    debug_pages_out = (
        output_dir
        /
        "debug"
        /
        "pages"
    )

    for directory in (
        images_out,
        labels_out,
        coco_out,
        transcription_out,
        review_out,
        debug_lines_out,
        debug_pages_out,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # COCO category mapping
    # --------------------------------------------------------

    category_by_id = {}

    for page in pages:

        for category in page[
            "coco_data"
        ].get(
            "categories",
            []
        ):

            category_by_id[
                int(
                    category["id"]
                )
            ] = category

    category_ids = sorted(
        category_by_id
    )

    yolo_class_by_category = {
        category_id: index
        for index, category_id
        in enumerate(
            category_ids
        )
    }

    line_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            category_by_id[
                category_id
            ]
            for category_id
            in category_ids
        ],
    }

    transcription_rows = []
    review_rows = []

    next_line_image_id = 1
    next_annotation_id = 1

    # ========================================================
    # Process pages
    # ========================================================

    for page_number, page in enumerate(
        pages,
        start=1
    ):

        page_id = page[
            "page_id"
        ]

        page_folder = page[
            "folder"
        ]

        print(
            f"\n[{page_number}/{len(pages)}] "
            f"Processing {page_id}"
        )

        lines, alto_size = read_alto(
            page[
                "alto"
            ]
        )

        if not lines:

            print(
                f"WARNING: no ALTO lines in {page_id}",
                file=sys.stderr
            )
            continue

        image_output_dir = (
            images_out
            /
            page_folder
        )

        label_output_dir = (
            labels_out
            /
            page_folder
        )

        debug_line_dir = (
            debug_lines_out
            /
            page_folder
        )

        for directory in (
            image_output_dir,
            label_output_dir,
            debug_line_dir,
        ):

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

        # ----------------------------------------------------
        # Get annotations for this page
        # ----------------------------------------------------

        coco_image_id = page[
            "coco_image"
        ][
            "id"
        ]

        page_annotations = [
            annotation
            for annotation
            in page[
                "coco_data"
            ].get(
                "annotations",
                []
            )
            if annotation.get(
                "image_id"
            )
            ==
            coco_image_id
        ]

        # ----------------------------------------------------
        # Assign each annotation to an ALTO line
        # ----------------------------------------------------

        assignments = {
            index: []
            for index
            in range(
                len(lines)
            )
        }

        for annotation in page_annotations:

            try:

                (
                    line_index,
                    distance,
                    centre_x,
                    centre_y
                ) = assign_annotation_to_line(
                    annotation,
                    lines
                )

            except ValueError as exc:

                print(
                    f"WARNING: annotation "
                    f"{annotation.get('id')} skipped: {exc}",
                    file=sys.stderr
                )

                continue

            assignments[
                line_index
            ].append(
                (
                    annotation,
                    distance,
                    centre_x,
                    centre_y
                )
            )

        # ----------------------------------------------------
        # Open source page
        # ----------------------------------------------------

        with Image.open(
            page[
                "image"
            ]
        ) as page_image:

            page_image = page_image.convert(
                "RGB"
            )

            (
                page_width,
                page_height
            ) = page_image.size

            # ------------------------------------------------
            # Page-level debug
            # ------------------------------------------------

            draw_page_debug(
                page_image=page_image,
                lines=lines,
                assignments=assignments,
                category_by_id=category_by_id,
                output_path=(
                    debug_pages_out
                    /
                    f"{page_folder}_assignment_debug.png"
                )
            )

            # =================================================
            # Process every ALTO line
            # =================================================

            for line_index, line in enumerate(
                lines
            ):

                # --------------------------------------------
                # Calculate enlarged crop
                # --------------------------------------------

                (
                    left,
                    top,
                    right,
                    bottom
                ) = compute_line_crop_bounds(
                    line=line,
                    assigned_annotations=assignments[
                        line_index
                    ],
                    page_width=page_width,
                    page_height=page_height,
                    horizontal_padding=horizontal_padding,
                    vertical_padding_ratio=vertical_padding_ratio,
                    polygon_padding=polygon_padding,
                    max_expand_ratio=max_expand_ratio,
                    crop_expand_distance=crop_expand_distance,
                )

                if (
                    right <= left
                    or
                    bottom <= top
                ):
                    continue

                crop_width = (
                    right - left
                )

                crop_height = (
                    bottom - top
                )

                line_filename = (
                    f"{line_index:04d}_"
                    f"{safe_name(line.line_id)}"
                )

                image_filename = (
                    f"{line_filename}.png"
                )

                label_filename = (
                    f"{line_filename}.txt"
                )

                # --------------------------------------------
                # Crop image
                # --------------------------------------------

                crop = page_image.crop(
                    (
                        left,
                        top,
                        right,
                        bottom
                    )
                )

                crop.save(
                    image_output_dir
                    /
                    image_filename
                )

                line_image_id = (
                    next_line_image_id
                )

                next_line_image_id += 1

                relative_image_path = (
                    Path(
                        page_folder
                    )
                    /
                    image_filename
                ).as_posix()

                # --------------------------------------------
                # Add line image to new COCO
                # --------------------------------------------

                line_coco[
                    "images"
                ].append(
                    {
                        "id":
                            line_image_id,

                        "file_name":
                            relative_image_path,

                        "width":
                            crop_width,

                        "height":
                            crop_height,

                        "source_page":
                            page[
                                "image"
                            ].name,

                        "alto_line_id":
                            line.line_id,

                        "transcription":
                            line.text,

                        "source_crop":
                            [
                                left,
                                top,
                                right,
                                bottom
                            ],
                    }
                )

                transcription_rows.append(
                    {
                        "page_id":
                            page_id,

                        "line_index":
                            line_index,

                        "line_id":
                            line.line_id,

                        "image":
                            relative_image_path,

                        "text":
                            line.text,
                    }
                )

                yolo_lines = []
                debug_polygons = []

                # ============================================
                # Process annotations assigned to this line
                # ============================================

                for (
                    annotation,
                    distance,
                    centre_x,
                    centre_y
                ) in assignments[
                    line_index
                ]:

                    category_id = int(
                        annotation[
                            "category_id"
                        ]
                    )

                    class_id = (
                        yolo_class_by_category.get(
                            category_id
                        )
                    )

                    if class_id is None:
                        continue

                    original_polygons = (
                        segmentation_polygons(
                            annotation
                        )
                    )

                    if not original_polygons:

                        review_rows.append(
                            {
                                "page_id":
                                    page_id,

                                "line_id":
                                    line.line_id,

                                "annotation_id":
                                    annotation.get(
                                        "id"
                                    ),

                                "category_id":
                                    category_id,

                                "baseline_distance":
                                    f"{distance:.2f}",

                                "status":
                                    "REVIEW_NO_POLYGON",
                            }
                        )

                        continue

                    transferred_polygons = []

                    # ========================================
                    # PAGE -> CROP coordinate conversion
                    # ========================================

                    for original_polygon in original_polygons:

                        # First make sure that no coordinate
                        # lies outside the final crop.
                        clipped_page_polygon = (
                            clip_polygon_to_rect(
                                original_polygon,
                                left,
                                top,
                                right,
                                bottom
                            )
                        )

                        if len(
                            clipped_page_polygon
                        ) < 6:
                            continue

                        # Convert absolute PAGE coordinates
                        # into local LINE-CROP coordinates.
                        local_polygon = (
                            translate_polygon_to_crop(
                                clipped_page_polygon,
                                left,
                                top
                            )
                        )

                        if (
                            polygon_area(
                                local_polygon
                            )
                            <=
                            0
                        ):
                            continue

                        transferred_polygons.append(
                            local_polygon
                        )

                        debug_polygons.append(
                            {
                                "polygon":
                                    local_polygon,

                                "category_id":
                                    category_id,

                                "annotation_id":
                                    annotation.get(
                                        "id"
                                    ),
                            }
                        )

                    if not transferred_polygons:

                        review_rows.append(
                            {
                                "page_id":
                                    page_id,

                                "line_id":
                                    line.line_id,

                                "annotation_id":
                                    annotation.get(
                                        "id"
                                    ),

                                "category_id":
                                    category_id,

                                "baseline_distance":
                                    f"{distance:.2f}",

                                "status":
                                    "REVIEW_OUTSIDE_CROP",
                            }
                        )

                        continue

                    # ----------------------------------------
                    # Recalculate bbox in crop coordinates
                    # ----------------------------------------

                    xs = [
                        x
                        for polygon
                        in transferred_polygons
                        for x
                        in polygon[0::2]
                    ]

                    ys = [
                        y
                        for polygon
                        in transferred_polygons
                        for y
                        in polygon[1::2]
                    ]

                    bbox = [
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ]

                    area = sum(
                        polygon_area(
                            polygon
                        )
                        for polygon
                        in transferred_polygons
                    )

                    # ----------------------------------------
                    # Add new crop-level COCO annotation
                    # ----------------------------------------

                    line_coco[
                        "annotations"
                    ].append(
                        {
                            "id":
                                next_annotation_id,

                            "image_id":
                                line_image_id,

                            "category_id":
                                category_id,

                            "segmentation":
                                transferred_polygons,

                            "bbox":
                                bbox,

                            "area":
                                area,

                            "iscrowd":
                                int(
                                    annotation.get(
                                        "iscrowd",
                                        0
                                    )
                                ),

                            "source_annotation_id":
                                annotation.get(
                                    "id"
                                ),
                        }
                    )

                    next_annotation_id += 1

                    # ----------------------------------------
                    # YOLO segmentation
                    # ----------------------------------------

                    # YOLO uses one polygon per object.
                    # If COCO contains multiple polygons,
                    # retain the largest one.
                    largest_polygon = max(
                        transferred_polygons,
                        key=polygon_area
                    )

                    normalized_polygon = (
                        normalize_polygon_for_yolo(
                            largest_polygon,
                            crop_width,
                            crop_height
                        )
                    )

                    yolo_lines.append(
                        str(
                            class_id
                        )
                        +
                        " "
                        +
                        " ".join(
                            f"{value:.6f}"
                            for value
                            in normalized_polygon
                        )
                    )

                    # ----------------------------------------
                    # Review status
                    # ----------------------------------------

                    status_flags = []

                    if distance > review_distance:

                        status_flags.append(
                            "BASELINE_DISTANCE"
                        )

                    if len(
                        transferred_polygons
                    ) > 1:

                        status_flags.append(
                            "MULTI_POLYGON_YOLO_LARGEST_ONLY"
                        )

                    if status_flags:

                        status = (
                            "REVIEW_"
                            +
                            "+".join(
                                status_flags
                            )
                        )

                    else:

                        status = "OK"

                    review_rows.append(
                        {
                            "page_id":
                                page_id,

                            "line_id":
                                line.line_id,

                            "annotation_id":
                                annotation.get(
                                    "id"
                                ),

                            "category_id":
                                category_id,

                            "baseline_distance":
                                f"{distance:.2f}",

                            "status":
                                status,
                        }
                    )

                # --------------------------------------------
                # Save YOLO labels
                # --------------------------------------------

                (
                    label_output_dir
                    /
                    label_filename
                ).write_text(
                    "\n".join(
                        yolo_lines
                    )
                    +
                    (
                        "\n"
                        if yolo_lines
                        else ""
                    ),
                    encoding="utf-8"
                )

                # --------------------------------------------
                # Save line-level debug image
                # --------------------------------------------

                draw_line_debug(
                    crop=crop,
                    debug_polygons=debug_polygons,
                    category_by_id=category_by_id,
                    output_path=(
                        debug_line_dir
                        /
                        f"{line_filename}_debug.png"
                    )
                )

        print(
            f"  ALTO lines: {len(lines)}"
        )

        print(
            f"  COCO annotations: {len(page_annotations)}"
        )

    # ========================================================
    # Save transcription CSV
    # ========================================================

    transcription_csv = (
        transcription_out
        /
        "transcriptions.csv"
    )

    with transcription_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "page_id",
                "line_index",
                "line_id",
                "image",
                "text",
            ]
        )

        writer.writeheader()

        writer.writerows(
            transcription_rows
        )

    # ========================================================
    # Save assignment review
    # ========================================================

    review_csv = (
        review_out
        /
        "assignment_review.csv"
    )

    with review_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "page_id",
                "line_id",
                "annotation_id",
                "category_id",
                "baseline_distance",
                "status",
            ]
        )

        writer.writeheader()

        writer.writerows(
            review_rows
        )

    # ========================================================
    # Save line-level COCO
    # ========================================================

    coco_json = (
        coco_out
        /
        "instances_lines.json"
    )

    coco_json.write_text(
        json.dumps(
            line_coco,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # Save YOLO classes
    # ========================================================

    class_file = (
        output_dir
        /
        "classes.txt"
    )

    class_lines = [
        (
            f"{yolo_class_by_category[category_id]}"
            "\t"
            f"{category_by_id[category_id].get('name', category_id)}"
        )
        for category_id
        in category_ids
    ]

    class_file.write_text(
        "\n".join(
            class_lines
        )
        +
        (
            "\n"
            if class_lines
            else ""
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Line crops:        {images_out}"
    )

    print(
        f"YOLO labels:       {labels_out}"
    )

    print(
        f"Line debug:        {debug_lines_out}"
    )

    print(
        f"Page debug:        {debug_pages_out}"
    )

    print(
        f"Line-level COCO:   {coco_json}"
    )

    print(
        f"Review CSV:        {review_csv}"
    )

    print(
        f"Transcriptions:    {transcription_csv}"
    )


# ============================================================
# Command line
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Crop ALTO TextLines and transfer "
            "page-level COCO letter polygons "
            "into line-level crops."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "input"
        ),
        help=(
            "Input directory containing "
            "images/, alto/, coco/. "
            "Default: ./input"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output"
        ),
        help=(
            "Output directory. "
            "Default: ./output"
        ),
    )

    parser.add_argument(
        "--horizontal-padding",
        type=int,
        default=10,
        help=(
            "Horizontal padding around each ALTO line. "
            "Default: 10 pixels"
        ),
    )

    parser.add_argument(
        "--vertical-padding-ratio",
        type=float,
        default=0.35,
        help=(
            "Vertical padding expressed as a fraction "
            "of ALTO line height. Default: 0.35"
        ),
    )

    parser.add_argument(
        "--polygon-padding",
        type=int,
        default=5,
        help=(
            "Extra pixels around assigned letter polygons. "
            "Default: 5"
        ),
    )

    parser.add_argument(
        "--max-expand-ratio",
        type=float,
        default=0.75,
        help=(
            "Maximum extra crop expansion relative to "
            "ALTO line height. Default: 0.75"
        ),
    )

    parser.add_argument(
        "--crop-expand-distance",
        type=float,
        default=40.0,
        help=(
            "Only annotations closer than this many pixels "
            "to the assigned baseline may enlarge the crop. "
            "Default: 40"
        ),
    )

    parser.add_argument(
        "--review-distance",
        type=float,
        default=25.0,
        help=(
            "Flag annotation assignments farther than this "
            "distance from the baseline. Default: 25 pixels"
        ),
    )

    args = parser.parse_args()

    pages = discover_pages(
        args.input.resolve()
    )

    print(
        f"Discovered {len(pages)} complete page(s)."
    )

    for page in pages:

        print(
            f"  {page['page_id']} -> "
            f"{page['image'].name} / "
            f"{page['alto'].name} / "
            f"{page['coco'].name}"
        )

    process_dataset(
        pages=pages,
        output_dir=args.output.resolve(),
        horizontal_padding=args.horizontal_padding,
        vertical_padding_ratio=args.vertical_padding_ratio,
        polygon_padding=args.polygon_padding,
        max_expand_ratio=args.max_expand_ratio,
        crop_expand_distance=args.crop_expand_distance,
        review_distance=args.review_distance,
    )


if __name__ == "__main__":
    main()
