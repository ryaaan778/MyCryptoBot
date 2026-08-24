"""Runtime inference for learned policies — numpy only, never torch.

The trading server must not import torch. A process holding real positions
should not be able to die because of a broken CUDA install, a version conflict,
or a 500MB dependency it uses for nothing at runtime. So training happens in
``research/`` and exports two artefacts: an SB3 ``model.zip`` for further
training, and a plain ``weights.npz`` that this package reads with a hand-written
forward pass.

A test asserts the two agree to 1e-6 on identical inputs. If they ever diverge,
the policy that trades is not the policy that was validated.
"""

from .numpy_infer import MLPPolicyWeights, NumpyPolicy, load_weights
from .loader import PolicyBundle, load_policy

__all__ = ["MLPPolicyWeights", "NumpyPolicy", "load_weights", "PolicyBundle", "load_policy"]
