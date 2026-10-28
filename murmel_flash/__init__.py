"""murmel-flash: Dutch + English speech recognition (FastConformer hybrid CTC/attention) with punctuation, word
timestamps and domain word lists.

    from murmel_flash import Transcriber, load_audio
"""
from .inference import DEFAULT_MODEL, Transcriber, load_audio
from .streaming import StreamingSession

__version__ = "4.1.0rc1"
__all__ = ["Transcriber", "StreamingSession", "load_audio", "DEFAULT_MODEL", "__version__"]
