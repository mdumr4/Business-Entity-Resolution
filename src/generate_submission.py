"""Inference CLI Entry Point.

Delegates to the unified, checkpointed inference pipeline in src.pipeline.predict.
"""

from src.pipeline.predict import main

if __name__ == "__main__":
    main()
