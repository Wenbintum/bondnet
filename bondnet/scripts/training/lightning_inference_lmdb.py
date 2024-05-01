import wandb, argparse, torch, json
import numpy as np
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from bondnet.data.datamodule import BondNetLightningDataModule, BondNetLightningDataModuleLMDB

from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    LogParameters,
    load_model_lightning,
)

seed_torch()
torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")
from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("-project_name", type=str, default="CRNs_0424")
    parser.add_argument("-run_name", type=str, default="tmp_run") #!wx
    parser.add_argument(
        "-dataset_loc", type=str, default="../../dataset/qm_9_merge_3_qtaim.json", help="dataset location, don't use if specifying LMDBs"
    )
    parser.add_argument("-log_save_dir", type=str, default=None)
    parser.add_argument("-config", type=str, default="./settings_lmdb.json")

    parser.add_argument(
        "--use_lmdb", default=True, action="store_true", help="use lmdbs"
    )

    parser.add_argument("-train_set", type=str, default="/pscratch/sd/w/wenxu/jobs/CRNs/data_2M/train_id_30") #!wx
    
    parser.add_argument("-test_set", type=str, required=True) #!wx
    parser.add_argument("-checkpoint", type=str, required=True)


    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    use_lmdb = bool(args.use_lmdb)
    project_name = args.project_name
    run_name = args.run_name #!wx
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    train_set = args.train_set
    test_set = args.test_set
    checkpoint = args.checkpoint
    config = args.config
    config = json.load(open(config, "r"))


    if config["model"]["precision"] == "16" or config["model"]["precision"] == "32":
        config["model"]["precision"] = int(config["model"]["precision"])

    # dataset
    extra_keys = config["model"]["extra_features"]
    config["model"]["filter_sparse_rxns"] = False
    config["model"]["debug"] = debug
    config["dataset"]["data_dir"] = dataset_loc
    config["dataset_transfer"]["data_dir"] = dataset_loc

    #!valset for multiple running.
    config["dataset"]["train_lmdb"] = train_set
    config["dataset"]["test_lmdb"] = test_set


    print("Using LMDB for dataset!...")
    dm = BondNetLightningDataModuleLMDB(config)

    feature_size, feature_names = dm.prepare_data()
    config["model"]["in_feats"] = feature_size
    config["dataset"]["feature_names"] = feature_names


    model = GatedGCNReactionNetworkLightning.load_from_checkpoint(
                checkpoint_path=checkpoint
            )
    
    print("model constructed!")
    #!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>for inference.
    #!change device, something
    config["optim"]["num_devices"] = 1
    config["optim"]["num_nodes"] = 1

    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="gpu",
        devices=config["optim"]["num_devices"],
        num_nodes=config["optim"]["num_nodes"],
        gradient_clip_val=config["optim"]["gradient_clip_val"],
        accumulate_grad_batches=config["optim"]["accumulate_grad_batches"],
        enable_progress_bar=True,
        enable_checkpointing=True,
        strategy=config["optim"]["strategy"],
        default_root_dir=config["dataset"]["log_save_dir"],
        precision=config["model"]["precision"],
    )


    trainer.test(model, dm)



