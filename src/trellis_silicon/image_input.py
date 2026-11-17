"""Open an input image the way viewers display it (issue #11).

Phones store pixels in sensor orientation and record the display rotation in the
EXIF ``Orientation`` tag. Without applying it, a portrait photo reaches the
pipeline lying on its side. Gradio's ``gr.Image(type="pil")`` already applies
the tag, so only the CLI needs this. Kept free of torch so it imports cheaply.
"""

from PIL import Image, ImageOps


def open_image(path):
    """Open ``path`` and apply its EXIF orientation. Untagged images are unchanged."""
    return ImageOps.exif_transpose(Image.open(path))
