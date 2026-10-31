"""Replace these two functions with the actual turntable/slider SDK calls.

Never pass this example to a real capture: it deliberately raises instead of
pretending that hardware moved. Positive degrees rotate forward; positive
millimeters move the camera/projector carriage upward.
"""


def rotate(degrees: float) -> None:
    raise NotImplementedError('Connect rotate(degrees) to the turntable API')


def slide(millimeters: float) -> None:
    raise NotImplementedError('Connect slide(millimeters) to the slider API')


def wait_idle() -> None:
    """Optional: block until both motors are stationary."""
    pass
