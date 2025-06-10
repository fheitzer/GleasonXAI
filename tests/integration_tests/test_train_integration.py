# tests/test_training.py
import sys
from pathlib import Path
import hydra
import pytest
import torch

# Make the repo root importable when the tests are run from inside /tests
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import train
# ------------------------------------------------------------------------------
# 1. Fixtures & helpers
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_dataset_cls():

    class Dummy(torch.utils.data.Dataset):
        def __init__(self, *_, **__):
            self.num_classes = 2      # Benign vs. Tumour – enough for the model & loss
            self.label_level = 0
            self.exp_numbered_lvl_remapping = [{0: [0], 1: [1]}]

        def __len__(self):
            return 4                 # tiny but > 0 for a DataLoader

        def __getitem__(self, idx):
            x  = torch.randn(3, 32, 32)              # image
            y  = torch.zeros(2, 32, 32)              # two-class one-hot mask
            y[torch.randint(0,2,()).item()] = 1
            ignore = torch.zeros(32, 32).bool()      # no ignored pixels
            return x, y, ignore
    return Dummy


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    """
    Provide fresh folders for EXPERIMENT_LOCATION
    """
    exp_dir  = tmp_path / "exp"
    exp_dir.mkdir()
    monkeypatch.setenv("EXPERIMENT_LOCATION", str(exp_dir))
    return exp_dir


def compose_test_cfg():
    """
    Load configs once, then override what is required
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()          # reset between tests
    with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT/"configs")):
        cfg = hydra.compose(config_name="config", overrides=["trainer.max_epochs=1",
                                                              "dataloader.num_workers=8",
                                                              "+trainer.limit_train_batches=3",
                                                              "+trainer.limit_val_batches=3",
                                                              "+trainer.limit_test_batches=3",
                                                              "dataloader.batch_size=1",
                                                              "dataloader.effective_batch_size=1",
                                                              "trainer.accelerator=cpu",
                                                              "loss_functions=cross_entropy",
                                                              "experiment=unit_test",
                                                              "logger.log_wandb=False",
                                                              ])

    return cfg

# ------------------------------------------------------------------------------
# 2. Tests
# ------------------------------------------------------------------------------

def test_train_end_to_end(monkeypatch, isolated_env, dummy_dataset_cls):
    """
    Calls train(cfg) directly with a with isolated save_dirs. Does not log to wandb.
    The test passes if the function runs without raising an exception and
    creates a checkpoint file in the expected location.
    Warning: this takes some time.
    """
    # --- patch dataset & model instantiation ---------------------------------
    #monkeypatch.setattr(gleasonxai.gleason_data, "GleasonX", dummy_dataset_cls)

    cfg = compose_test_cfg()

    # Training passes but takes ages.

    # -------------------------------------------------------------------------
    train(cfg)                            # the call must *not* raise
    assert isolated_env/cfg.experiment/"version_0"/"checkpoints"/"best_model.ckpt".exists()
    