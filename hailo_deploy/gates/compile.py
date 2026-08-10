"""Re-export dummy compile probe; real HEF compile is ``compile_real``."""
from hailo_deploy.compile import *  # noqa: F401,F403
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: F401
