"""Small adapter for the two signed relative-motion hardware APIs.

An adapter module exports rotate(degrees), slide(millimeters), and optionally
wait_idle(). Positive rotation is forward; positive slide is upward. Calls
must return only after motion has stopped, or provide wait_idle().
"""
import importlib.util
from pathlib import Path


class MotionController:
    def __init__(self, module_path):
        path = Path(module_path).resolve()
        spec = importlib.util.spec_from_file_location('scan_motion_adapter', path)
        if spec is None or spec.loader is None:
            raise ValueError(f'Cannot load motion adapter: {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ('rotate', 'slide'):
            if not callable(getattr(module, name, None)):
                raise ValueError(f'Motion adapter must define {name}(signed_amount)')
        self.module = module

    def _move(self, name, amount):
        getattr(self.module, name)(float(amount))
        wait = getattr(self.module, 'wait_idle', None)
        if callable(wait):
            wait()

    def rotate(self, degrees):
        self._move('rotate', degrees)

    def slide(self, millimeters):
        self._move('slide', millimeters)
