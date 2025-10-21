import logging
import os
from typing import Tuple

import cv2
import numpy as np

from .. import utils
from . import DATASET_ROOT_DIR

logger = logging.getLogger(__name__)


class FileTypeNotSupportedError(Exception):
    def __init__(self, message):
        logger.error(message)
        super().__init__(message)

class DataLoaderBase(object):
    """Base of the DataLoader class.
    Please make sure to implement
     - load_event()
     - get_sequence()
    in chile class.
    """

    NAME = "example"

    def __init__(self, config: dict = {}):
        # Cache related
        self.min_ts, self.max_ts = None, None
        self.clear_time_cache()
        self.clear_len_cache()
        self.clear_imu_cache()
        self.clear_pose_cache()
        self.clear_trigger_cache()

        # Basic configurations
        self._HEIGHT = config["height"]
        self._WIDTH = config["width"]
        root_dir: str = config["root"] if config["root"] else DATASET_ROOT_DIR
        self.root_dir: str = os.path.expanduser(root_dir)
        data_dir: str = config["dataset"] if config["dataset"] else self.NAME
        self.dataset_dir: str = os.path.join(self.root_dir, data_dir)
        self.__dataset_files: dict = {}
        logger.info(f"Loading directory in {self.dataset_dir}")


        # For differnt file format for differnt data        
        self.event_file_format = None
        self.frame_file_format = None
        self.depth_file_format = None
        self.pose_file_format = None
        self.trigger_file_format = None

        self.check_additional_data_load(config)

        if utils.check_key_and_bool(config, "undistort"):
            logger.info("Undistort events when load_event.")
            self.auto_undistort = True
        else:
            logger.info("No undistortion.")
            self.auto_undistort = False

    def check_additional_data_load(self, config):
        """Add additional data loading configuration: GT flow, GT pose, trigger, frame
        Be careful that it does NOT assume the availability of the additional data.

        Args:
            config (_type_): _description_
        """
        self.gt_flow_available = False
        self.gt_pose_available = False
        self.trigger_available = False
        self.frame_available = False

        self.trigger_try_to_load = utils.check_key_and_bool(config, "load_trigger")
        self.frame_try_to_load = utils.check_key_and_bool(config, "load_frame")
        self.gt_flow_try_to_load = utils.check_key_and_bool(config, "load_gt_flow")
        # This is for legacy API to support customized gt directories
        if "gt" in config.keys():
            self.gt_flow_dir = os.path.expanduser(config["gt"])
        else:
            self.gt_flow_dir = None
        self.gt_pose_try_to_load = utils.check_key_and_bool(config, "load_gt_pose")
        if "gt_pose" in config.keys():
            self.gt_pose_dir = os.path.expanduser(config["gt_pose"])
        else:
            self.gt_pose_dir = None

    def __len__(self):
        if self._len_cache is None:
            self.set_len_cache()
        return self._len_cache

    @property
    def dataset_files(self) -> dict:
        return self.__dataset_files

    @dataset_files.setter
    def dataset_files(self, sequence: dict):
        self.__dataset_files = sequence

    # Cache related.
    @property
    def t_min(self):
        if self.min_ts is None:
            if self._time_cache is None:
                self.set_time_cache()
            self.min_ts = self._time_cache.min()
        return self.min_ts

    @property
    def t_max(self):
        if self.max_ts is None:
            if self._time_cache is None:
                self.set_time_cache()
            self.max_ts = self._time_cache.max()
        return self.max_ts

    @property
    def duration(self):
        return self.t_max - self.t_min

    def set_len_cache(self):
        raise NotImplementedError

    def set_time_cache(self):
        raise NotImplementedError

    def set_trigger_cache(self):
        raise NotImplementedError

    def clear_time_cache(self):
        self._time_cache = None

    def clear_len_cache(self):
        self._len_cache = None

    def clear_pose_cache(self):
        self._pose_cache = None

    def clear_imu_cache(self):
        self._imu_cache = None

    def clear_trigger_cache(self):
        self._trigger_cache = None

    def set_sequence(self, sequence_name: str) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        self.dataset_files = self.get_sequence(sequence_name)
        self.post_set_sequence()
        logger.info(
            f"{self.gt_flow_available = }, {self.gt_pose_available = }, {self.trigger_available = }, {self.frame_available = }"
        )

    def post_set_sequence(self):
        pass

    def get_sequence(self, sequence_name: str) -> dict:
        raise NotImplementedError

    def load_event(
        self, start_index: int, end_index: int, cam: str = "left", *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def load_calib(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def load_calib_our_style(self, matrix_file: str, distortion_file: str) -> dict:
        """Load calibration file, produced by the calibrator script.

        Args:
            matrix_file (str): .npy file path
            distortion_file (str): .npy file path

        Returns:
            dict: _description_
        """
        K = np.load(matrix_file)
        D = np.load(distortion_file)
        # imshape = (self._WIDTH, self._HEIGHT)
        # newcameramtx, _ = cv2.getOptimalNewCameraMatrix(K, D, imshape, 1, imshape)
        # return {"K": newcameramtx, "D": D}
        return {"K": K, "D": D}

    def load_homography(self) -> np.ndarray:
        raise NotImplementedError

    def load_trigger(self) -> np.array:
        raise NotImplementedError

    def load_optical_flow(self, t1: float, t2: float, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def load_image(self, index: int) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def index_to_time(self, index: int) -> float:
        """Event index to time.
        Args:
            index (int): index of event
        Returns:
            float: time in sec
        """
        if self._time_cache is None:
            self.set_time_cache()
        return self._time_cache[index]

    def time_to_index(self, time: float, *args, **kwargs) -> int:
        """Time to event index.
        Args:
            time (float): time in sec
        Returns:
            int: index
        """
        if self._time_cache is None:
            self.set_time_cache()
        ind = np.searchsorted(self._time_cache, time)
        return ind - 1

    def warp_homography(self, event: np.ndarray) -> np.ndarray:
        """Warp events according to homography.
        Args:
            event (np.ndarray): [N, 4]. height, width, time, pol
        Returns:
            warped_event (np.ndarray): [N, 4]
        """
        homography = self.load_homography()
        return utils.warp_events_homography(event, homography)

    def free_up_flow(self):
        pass

    def setup_gt_flow(self):
        pass
