"""Entry point of the standalone app built by packaging/build.py."""

import multiprocessing
import sys

from brainstudio.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
