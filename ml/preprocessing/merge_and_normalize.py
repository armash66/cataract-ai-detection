"""Compatibility wrapper. Use ml.data.merge_and_normalize instead."""
from ml.data.merge_and_normalize import *  # noqa: F401,F403

if __name__ == "__main__":
    from ml.data.merge_and_normalize import main
    main()
