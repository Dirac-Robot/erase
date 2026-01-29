from ato.scope import Scope
from ato.adict import ADict

scope = Scope()


@scope.observe(default=True)
def default(config: ADict):
    config.model = 'gpt-4o-mini'
    config.threshold.retention = 0.5
    config.threshold.erasure = 0.7


@scope.observe
def strict(config: ADict):
    """Stricter thresholds for more aggressive memory pruning."""
    config.threshold.retention = 0.7
    config.threshold.erasure = 0.5


@scope.observe
def lenient(config: ADict):
    """Lenient thresholds to keep more memories."""
    config.threshold.retention = 0.3
    config.threshold.erasure = 0.9
