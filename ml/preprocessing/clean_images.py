"""Compatibility wrapper. Use ml.data.clean_images instead."""
from ml.data.clean_images import *  # noqa: F401,F403

if __name__ == "__main__":
    from ml.data.clean_images import main
    main()
