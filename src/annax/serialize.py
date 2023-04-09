import io
import pickle

import zstandard as zstd


def dump(obj, file_obj):
    bytes = pickle.dumps(obj)
    compressed = zstd.compress(bytes, level=3)
    file_obj.write(compressed)


def load(file_obj):
    compressed = file_obj.read()
    bytes = zstd.decompress(compressed)
    return pickle.loads(bytes)
