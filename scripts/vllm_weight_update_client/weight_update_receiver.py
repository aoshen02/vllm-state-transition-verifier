"""vLLM worker extension that applies ``receiver_patches`` in every worker.

    PYTHONPATH=scripts/vllm_weight_update_client vllm serve <model> \\
        --worker-extension-cls weight_update_receiver.WeightUpdateReceiver ...
"""

import receiver_patches

receiver_patches.apply()


class WeightUpdateReceiver:
    """Adds no worker methods: importing this module patches vLLM."""
