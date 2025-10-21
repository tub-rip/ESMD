# type: ignore
import argparse
import logging
import os
import shutil
import sys
from time import time

import numpy as np
import yaml
from tqdm import tqdm

from src import data_loader, solver, utils, visualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/denoise/cmax_denoise_davis.yaml",
        help="Config file yaml path",
        type=str,
    )
    parser.add_argument(
        "--log", help="Log level: [debug, info, warning, error, critical]", type=str, default="info"
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    return config, args


def save_config(save_dir: str, file_name: str, log_level=logging.INFO):
    """Save configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(file_name, save_dir)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"{save_dir}/main.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def sequential_withou_gt(data_config, loader, solv):
    logger.info("Just sequential evaluation")
    n_events = data_config["n_events_per_batch"]
    for i_frame, ind1 in tqdm(enumerate(range(0, len(loader) - n_events, n_events // 2))):
        try:
            if i_frame < data_config["ind1"] or i_frame > data_config["ind2"]:
                continue  # cutofff
        except KeyError:
            pass
        ind2 = ind1 + n_events
        batch = loader.load_event(ind1, ind2)
        batch[..., 2] -= np.min(batch[..., 2])
        timescale = batch[..., 2].max() - batch[..., 2].min()

        batch = solv.preprocess(batch)
        best_motion = solv.optimize(batch)  # this is per unit time. Need time scale
        solv.set_previous_frame_best_estimation(best_motion)

        t1 = loader.index_to_time(ind1)
        t2 = loader.index_to_time(ind2)
        fwl = solv.calculate_fwl_pred(best_motion, batch, timescale=timescale)  # type: ignore
        solv.save_flow_error_as_text(i_frame, fwl, "fwl_per_frame.txt")  # type: ignore

        solv.save_flow_error_as_text(
            i_frame, {"t1": t1, "t2": t2}, "timestamps_per_frame.txt"
        )  # save timestamp.txt
        solv.save_flow_error_as_text(
            i_frame, {"i1": ind1, "i2": ind2}, "indices_per_frame.txt"
        )  # save timestamp.txt
        solv.visualize_original_sequential(batch)
        solv.visualize_pred_sequential(batch, best_motion)


if __name__ == "__main__":
    config, args = parse_args()
    data_config: dict = config["data"]
    out_config: dict = config["output"]
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    save_config(out_config["output_dir"], args.config_file, log_level)
    logger = logging.getLogger(__name__)

    if utils.check_key_and_bool(config, "fix_random_seed"):
        utils.fix_random_seed()

    # Visualizer
    image_shape = (data_config["height"], data_config["width"])
    if config["is_dnn"] and "crop" in data_config["preprocess"].keys():
        image_shape = (data_config["preprocess"]["crop"]["height"], data_config["preprocess"]["crop"]["width"])  # type: ignore

    viz = visualizer.Visualizer(
        image_shape,
        show=out_config["show_interactive_result"],
        save=True,
        save_dir=out_config["output_dir"],
    )

    # Solver
    method_name = config["solver"]["method"]
    _load = data_loader.collections[data_config["dataset"]](config=data_config)
    _load.set_sequence(data_config["sequence"])
    solv: solver.SolverBase = solver.collections[method_name](
        image_shape,
        calibration_parameter=_load.load_calib(),
        solver_config=config["solver"],
        optimizer_config=config["optimizer"],
        output_config=config["output"],
        visualize_module=viz,
    )


    logger.info("Single-frame denoising")
    loader = data_loader.collections[data_config["dataset"]](config=data_config)
    loader.set_sequence(data_config["sequence"])

    ind1, ind2 = data_config["ind1"], data_config["ind2"]
    batch: np.ndarray = loader.load_event(ind1, ind2)
    batch[..., 2] -= np.min(batch[..., 2])
    batch = solv.preprocess(batch)

    best_motion, noise_probability = solv.denoise(batch)
    solv.visualize_denoise(batch, best_motion)
