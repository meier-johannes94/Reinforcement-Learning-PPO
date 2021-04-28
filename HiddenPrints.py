import sys
import os


class HiddenPrints:
    """Blocks print statements in a with block

        Example: 
        All Prints that would normally be done by operation() are now
        blocked. 

        with HiddenPrints():
            operation()
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
