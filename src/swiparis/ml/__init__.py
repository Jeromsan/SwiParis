"""Cœur ML de SwiParis : features φ(i), profil de session, choix des paires, classement.

Les noms ci-dessous sont chargés à la demande : `import swiparis.ml.evaluation`
(bibliothèque standard seulement) fonctionne donc sans numpy.
"""
from importlib import import_module

_EXPORTS = {
    "EncodedItems": "swiparis.ml.items",
    "FeatureSpace": "swiparis.ml.features",
    "Item": "swiparis.ml.features",
    "SessionProfile": "swiparis.ml.model",
    "encode_common": "swiparis.ml.items",
    "next_pair": "swiparis.ml.selection",
    "rank": "swiparis.ml.selection",
    "session_finished": "swiparis.ml.selection",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'swiparis.ml' has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
