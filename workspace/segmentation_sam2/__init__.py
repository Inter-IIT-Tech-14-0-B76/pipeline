
"""segmentation_sam2 package initializer.

This module ensures the bundled `sam2` package (located at
`segmentation_sam2/sam2`) can be imported using the top-level name
``sam2``. Many files inside the bundled SAM2 code use absolute imports
like ``from sam2.modeling...``. To make those resolve when the project
is run from the repository root, we add the package directory to
``sys.path`` so Python can find the nested ``sam2`` package as a
top-level package.

This is a minimal compatibility shim to avoid editing many internal
files in the SAM2 bundle.
"""

import os
import sys
import importlib

# Expose submodules through package exports
__all__ = ["prompt", "init"]

# Insert the package directory onto sys.path so `import sam2` finds
# the bundled `sam2` directory inside this package.
_pkg_dir = os.path.dirname(__file__)
if _pkg_dir not in sys.path:
	sys.path.insert(0, _pkg_dir)

# If possible, import the bundled sam2 module and also register it as
# the top-level "sam2" module in sys.modules so code that does
# `import sam2` or `from sam2 ...` will resolve to the bundled copy.
try:
	_sam2 = importlib.import_module(".sam2", package=__name__)
	sys.modules.setdefault("sam2", _sam2)
except Exception:
	# If import fails here, defer resolution until runtime — adding
	# _pkg_dir to sys.path above should allow normal imports later.
	pass
