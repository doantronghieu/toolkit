from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel as PBaseModel, Field
from torch.utils.data import DataLoader
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.runner import Runner, BaseLoop
from mmengine.hooks import Hook
from mmengine.registry import LOOPS, HOOKS, RUNNERS, MODELS, METRICS
from mmengine.logging import MessageHub
from mmengine.config import Config
from mmengine.visualization import Visualizer

class RunnerConfig(PBaseModel):
    project_name: str
    work_dir: str
    seed: int
    model: Dict[str, Any]
    train_dataloader: Dict[str, Any]
    val_dataloader: Dict[str, Any]
    train_cfg: Dict[str, Any]
    val_cfg: Dict[str, Any]
    val_evaluator: Dict[str, Any]
    optim_wrapper: Dict[str, Any]
    param_schedulers: Dict[str, Any]
    hooks: List[Dict[str, Any]]
    feature_flags: Dict[str, bool]
    checkpoint: Dict[str, Any]
    visualizer: Optional[Dict[str, Any]] = Field(default_factory=dict)
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    cudnn_benchmark: bool = False
    mp_start_method: str = 'fork'
    dist_params: Dict[str, Any] = Field(default_factory=lambda: {'backend': 'nccl'})
    log: Dict[str, Any] = Field(default_factory=dict)
    log_level: str = 'INFO'
    default_scope: str = 'mmengine'
    log_processor: Dict[str, Any] = Field(default_factory=dict)
    default_hooks: Dict[str, Any] = Field(default_factory=dict)
    launcher: str = 'none'
    env_cfg: Dict[str, Any] = Field(default_factory=dict)
    resume: bool = False  # New field for resume functionality
    cfg: Dict

class RunnerManager:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            self.config = Config.fromfile(config)
        else:
            self.config = Config(config)
        self.runner_config = RunnerConfig(**self.config)
        
        self.runner = self._build_runner()
        self.message_hub = MessageHub.get_current_instance()

    def _build_runner(self) -> Runner:
        """Build and return a runner based on the configuration."""
        return Runner(
            model=self.build_model(),
            work_dir=self.runner_config.work_dir,
            train_dataloader=self.build_dataloader(self.runner_config.train_dataloader, is_train=True),
            val_dataloader=self.build_dataloader(self.runner_config.val_dataloader, is_train=False),
            test_dataloader=None,  # Not provided in the config, add if needed
            train_cfg=self.runner_config.train_cfg,
            val_cfg=self.runner_config.val_cfg,
            test_cfg=None,  # Not provided in the config, add if needed
            auto_scale_lr=None,  # Not provided in the config, add if needed
            optim_wrapper=self.runner_config.optim_wrapper,
            param_scheduler=self.runner_config.param_schedulers,
            val_evaluator=self.build_evaluator(self.runner_config.val_evaluator),
            test_evaluator=None,  # Not provided in the config, add if needed
            default_hooks=self.runner_config.default_hooks,
            custom_hooks=self.runner_config.hooks,
            data_preprocessor=self.runner_config.model.get('data_preprocessor'),
            load_from=self.runner_config.load_from,
            resume=self.runner_config.resume,
            launcher=self.runner_config.launcher,
            env_cfg=self.runner_config.env_cfg,
            log_processor=self.runner_config.log_processor,
            log_level=self.runner_config.log_level,
            visualizer=self.runner_config.visualizer,  # Pass the visualizer config directly
            default_scope=self.runner_config.default_scope,
            randomness={'seed': self.runner_config.seed},
            experiment_name=self.runner_config.project_name,
            cfg=self.runner_config.cfg
        )
    def train(self) -> None:
        """Execute the training process."""
        self.runner.train()

    def validate(self) -> None:
        """Execute the validation process."""
        self.runner.val()

    def test(self) -> None:
        """Execute the testing process."""
        self.runner.test()

    def configure_logging(self, custom_cfg: Optional[List[Dict[str, Any]]] = None) -> None:
        """Configure logging for the runner."""
        if custom_cfg is not None:
            self.runner_config.log_processor['custom_cfg'] = custom_cfg
        self.runner.logger.setLevel(self.runner_config.log_level)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint for the model."""
        self.runner.load_checkpoint(checkpoint_path)

    def save_checkpoint(self, save_path: str) -> None:
        """Save a checkpoint of the current model state."""
        self.runner.save_checkpoint(save_path)

    def register_custom_hook(self, hook: Union[Hook, Dict[str, Any]]) -> None:
        """Register a custom hook."""
        self.runner.register_hook(hook)

    def register_custom_loop(self, loop_name: str, loop: Union[BaseLoop, Dict[str, Any]]) -> None:
        """Register a custom loop."""
        LOOPS.register_module(name=loop_name, module=loop)

    def update_custom_log(self, name: str, value: Any) -> None:
        """Update a custom log value."""
        self.message_hub.update_scalar(f'{self.runner.mode}/{name}', value)

    @staticmethod
    def register_custom_runner(runner_name: str, runner_class: Type[Runner]) -> None:
        """Register a custom runner class."""
        RUNNERS.register_module(name=runner_name, module=runner_class)

    def build_model(self) -> BaseModel:
        """Build and return a model based on the configuration."""
        return MODELS.build(self.runner_config.model)
    
    def build_dataloader(self, dataloader_cfg: Dict[str, Any], is_train: bool = True) -> DataLoader:
        """Build and return a dataloader based on the configuration."""
        return Runner.build_dataloader(
            dataloader_cfg,
            seed=self.runner_config.seed,
            diff_rank_seed=is_train
        )

    def build_evaluator(self, evaluator_cfg: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[BaseMetric, List[BaseMetric]]:
        """Build and return an evaluator based on the configuration."""
        if isinstance(evaluator_cfg, dict):
            return Evaluator(METRICS.build(evaluator_cfg))
        elif isinstance(evaluator_cfg, list):
            return [Evaluator(METRICS.build(cfg)) for cfg in evaluator_cfg]
        else:
            raise TypeError(f"Unsupported evaluator config type: {type(evaluator_cfg)}")
    
    def set_feature_flags(self) -> None:
        """Set feature flags based on the configuration."""
        for flag, value in self.runner_config.feature_flags.items():
            setattr(self.runner, flag, value)

    def configure_optimizers(self) -> None:
        """Configure optimizers based on the configuration."""
        self.runner.optim_wrapper.optimizer = RUNNERS.build(self.runner_config.optim_wrapper['optimizer'])

    def configure_schedulers(self) -> None:
        """Configure parameter schedulers based on the configuration."""
        self.runner.param_scheduler = RUNNERS.build(self.runner_config.param_schedulers)

    def configure_hooks(self) -> None:
        """Configure hooks based on the configuration."""
        for hook_cfg in self.runner_config.hooks:
            self.runner.register_hook(HOOKS.build(hook_cfg))

    def configure_checkpoint(self) -> None:
        """Configure checkpoint saving based on the configuration."""
        self.runner.default_hooks['checkpoint'].interval = self.runner_config.checkpoint['interval']

    def configure_distributed(self) -> None:
        """Configure distributed training based on the configuration."""
        if self.runner_config.launcher != 'none':
            self.runner.launcher = self.runner_config.launcher
            self.runner.distributed = True

    def run(self, tasks: List[str] = ['train', 'val', 'test']) -> None:
        """Run the specified tasks in the pipeline."""
        self.set_feature_flags()
        self.configure_optimizers()
        self.configure_schedulers()
        self.configure_hooks()
        self.configure_checkpoint()
        self.configure_distributed()
        
        for task in tasks:
            if task == 'train':
                self.train()
            elif task == 'val':
                self.validate()
            elif task == 'test':
                self.test()
            else:
                raise ValueError(f"Unsupported task: {task}")
