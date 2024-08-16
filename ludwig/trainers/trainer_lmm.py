#imports here
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry
from ludwig.api_annotations import DeveloperAPI

#decide whether these things go under schema/trainer.py or new lmm specific file?
#trainer.py includes GBM/ECD/LLM params, so i think it makes sense to separate each trainer into its own file

@DeveloperAPI
def register_lmm_trainer(trainer_type: str):
    def wrap(trainer_confiig: BaseTrainerConfig):
        trainer_schema_registry[model_type] = trainer_config
        return trainer_confiig
    return wrap

#going to skip NoneTrainer for now, only build out finetuner
@register_lmmm_trainer("finetune")
@ludwig_dataclass
class FineTuneTrainerConfig(LMMTrainerConfig):
    pass