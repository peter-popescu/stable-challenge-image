"""Fail before experiment writes if the destination filesystem is nearly full."""
from pathlib import Path
import shutil


MINIMUM_FREE_BYTES = 8 * 1024**3


def require_free_space(destination, minimum_bytes=MINIMUM_FREE_BYTES):
    """Check the destination's existing ancestor, following mounted symlinks."""
    target = Path(destination).resolve()
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < minimum_bytes:
        raise OSError(
            f'Insufficient free storage for {target}: '
            f'{free / 1024**3:.2f} GiB available on {ancestor}; '
            f'at least {minimum_bytes / 1024**3:.2f} GiB required before '
            'experiment writes. Keep jobs held until an approved storage '
            'location with sufficient capacity is configured. No files '
            'have been deleted or relocated by this guard.'
        )
    return dict(destination=str(target), filesystem_probe=str(ancestor),
                free_bytes=free, minimum_free_bytes=minimum_bytes)
