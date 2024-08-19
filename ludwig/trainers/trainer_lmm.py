#imports here
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry
from ludwig.api_annotations import DeveloperAPI
from ludwig.trainers.trainer import Trainer
from ludwig.trainers.registry import register_lmm_trainer
from ludwig.schema.trainer import FineTuneTrainerConfig
from ludwig.models.lmm import LMM
from typing import List, Optional
from ludwig.utils.torch_utils import get_torch_device
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.utils.defaults import default_random_seed

#decide whether these things go under schema/trainer.py or new lmm specific file?
#trainer.py includes GBM/ECD/LLM params, so i think it makes sense to separate each trainer into its own file

#going to skip NoneTrainer for now, only build out finetuner
@register_lmm_trainer("finetune")
class FineTuneTrainerConfig(Trainer):
    @staticmethod
    def get_schema_cls():
        return FineTuneTrainerConfig
    
    def __init__( 
        self,
        config: FineTuneTrainerConfig,
        model: LMM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        report_tqdm_to_ray=False,
        random_seed: int = default_random_seed,
        distributed: Optional[DistributedStrategy] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            config,
            model,
            resume,
            skip_save_model,
            skip_save_progress,
            skip_save_log,
            callbacks,
            report_tqdm_to_ray,
            random_seed,
            distributed,
            device,
            **kwargs,
        )
    