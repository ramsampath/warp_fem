import sys
print("Python:", sys.version)

try:
    import warp as wp
    print("Warp version:", getattr(wp, "__version__", "unknown"))
    print("Warp module:", wp.__file__)
except Exception as e:
    print("Warp import error:", e)
