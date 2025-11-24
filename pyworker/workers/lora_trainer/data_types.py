import dataclasses
import inspect
import sys
import os
from typing import Dict, Any

from lib.data_types import ApiPayload, JsonDataException

@dataclasses.dataclass
class LoraJobPayload(ApiPayload):
    """
    Defines the input data for a LoRA training job.
    This is the payload sent from the client (our main app's BullMQ worker).
    """
    job_id: str
    user_id: int
    image_r2_key: str
    webhook_url: str

    @classmethod
    def for_test(cls) -> "LoraJobPayload":
        """Generates dummy data for testing, e.g., for benchmarks."""
        return cls(
            job_id="test_job_12345",
            user_id=0,
            image_r2_key="test/image.jpg",
            webhook_url="https://example.com/webhook"
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        """
        This PyWorker doesn't call a model API directly, so this method is not strictly necessary,
        but it's implemented to satisfy the abstract class requirements.
        """
        return dataclasses.asdict(self)

    def count_workload(self) -> float:
        """
        Defines the workload for a LoRA training job.
        We assume a constant load of 100 per job. Vast.ai uses this value for scaling decisions.
        """
        return 100.0

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "LoraJobPayload":
        """
        Transforms the JSON payload received from the client into a LoraJobPayload object.
        """
        # Check for missing required parameters
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg and cls.__dataclass_fields__[param].default is dataclasses.MISSING:
                 errors[param] = f"missing required parameter: {param}"
        
        if errors:
            raise JsonDataException(errors)
            
        # Ignore unknown parameters and instantiate with known ones
        known_params = {k: v for k, v in json_msg.items() if k in inspect.signature(cls).parameters}
        return cls(**known_params)
