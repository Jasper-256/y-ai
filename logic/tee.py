"""Duplicate text output to several streams (Unix tee-style).

Example:
    with open("results.txt", "w") as out_f:
        out = Tee(sys.stdout, out_f)
        print("...", file=out)
"""

class Tee:
    """Mirror write/flush to several text streams (for `print(..., file=...)`)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()
