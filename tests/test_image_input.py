"""The CLI applies EXIF orientation to input photos (issue #11)."""

from PIL import Image

from trellis_silicon.image_input import open_image


def _save(path, orientation=None, mode="RGB"):
    # 4 wide x 2 tall, left half red, right half blue.
    im = Image.new(mode, (4, 2), "blue")
    im.paste("red", (0, 0, 2, 2))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    im.save(path, exif=exif)
    return im


def test_orientation_6_is_rotated_upright(tmp_path):
    path = tmp_path / "phone.png"
    _save(path, orientation=6)
    raw = Image.open(path)
    assert raw.size == (4, 2) and raw.getexif().get(274) == 6

    out = open_image(path)
    # Orientation 6: rotate 90 deg clockwise for display, so red moves to the top.
    assert out.size == (2, 4)
    assert out.getpixel((0, 0)) == (255, 0, 0)
    assert out.getpixel((1, 3)) == (0, 0, 255)
    assert out.getexif().get(274) is None


def test_untagged_image_is_unchanged(tmp_path):
    path = tmp_path / "plain.png"
    src = _save(path, mode="RGBA")
    out = open_image(path)
    assert out.size == src.size
    assert out.mode == "RGBA"
    assert out.tobytes() == src.tobytes()
