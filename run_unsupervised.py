from omegaconf import DictConfig
from src.unsupervised.unsupervised_eval import eval_unsup
import hydra

@hydra.main(config_path="./configs/unsupervised", config_name="config")
def eval_unsupervised(cfg: DictConfig) -> None:   
    print(cfg.experiment)
    eval_run = hydra.utils.instantiate(cfg.experiment)
    eval_unsup(eval_run)

if __name__ == "__main__":   
    eval_unsupervised()