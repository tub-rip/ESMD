import logging
import os

import numpy as np

from . import DataLoaderBase, DavisDataLoader, fileformat, FileTypeNotSupportedError
from ..utils import check_file_utils, infer_event_timestamps_into_sec

logger = logging.getLogger(__name__)


class EndDataLoader(DavisDataLoader):
    """Dataloader class for Denoising.
    https://github.com/KugaMaxx/cuke-emlb
    """

    NAME = "END"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.gt_flow_available = False  # always false
        self.frame_available = True
        self.dn_type = config["type"].upper()
        self.sequence_detail = config["detail"]
    
    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `outdoot_day2`.

        Returns:
            sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        # data_path = os.path.join(self.dataset_dir, sequence_name)
        data_path: str = os.path.join(self.root_dir, "END", f"{self.dn_type}-END", sequence_name)
        calib_file: str = os.path.join(self.root_dir, "END", f"{self.dn_type}-END", ".Calib" f"{sequence_name}.xml")

        event_aedat_file = os.path.join(data_path, f"{sequence_name}-{self.sequence_detail}.aedat4")
        event_cache_file = event_aedat_file.replace("aedat4", "hdf5")

        if check_file_utils(event_cache_file):
            self.event_file_format = "hdf5"
        elif check_file_utils(event_aedat_file):
            self.event_file_format = "aedat"
        else:
            raise FileTypeNotSupportedError(f"Cannot find {event_cache_file} or {event_aedat_file}.")
        # Cache
        if self.event_file_format != "hdf5" and self._make_hdf5_cache:
            if self.event_file_format == "aedat":
                ev_iter = fileformat.IteratorAedat4Event(event_aedat_file)
            else:
                raise FileTypeNotSupportedError(f"{self.event_file_format} is not supported.")
            data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}  # type: ignore
            fileformat.convert_iterator_access_to_hdf5(ev_iter, event_cache_file, data_keys)
            self.event_file_format = "hdf5"
        logger.info(f"Event data format: {self.event_file_format}")

        # Frames
        frame_cache_file = event_cache_file.replace(".hdf5", "_frames.hdf5")  # hdf5 file support
        if check_file_utils(frame_cache_file):
            self.frame_file_format = "hdf5"
        elif check_file_utils(event_aedat_file):
            self.frame_file_format = "aedat"  # frame is encoded with event files..
        # Cache
        if self.frame_file_format != "hdf5" and self._make_hdf5_cache:
            if self.frame_file_format == "aedat":
                fr_iter = fileformat.IteratorAedat4Frame(event_aedat_file)
            data_keys = {"frame": "frames/raw", "t": "frames/t"}  # type: ignore
            fileformat.convert_iterator_access_to_hdf5(
                fr_iter, frame_cache_file, data_keys, image_keys=["frame"]
            )
            self.frame_file_format = "hdf5"
        logger.info(f"Frame data format: {self.frame_file_format}")

        sequence_file = {
            "event_dat": event_aedat_file,  # dat
            "event_cache": event_cache_file,  # hdf5
            "frame_cache": frame_cache_file,  # hdf5
            "calib": calib_file,
        }
        return sequence_file

    def post_set_sequence(self):
        if self.event_file_format == "hdf5":
            logger.info("Optimize data loading speed. Loading events...")
            self.preloaded_event = fileformat.load_hdf5(
                self.dataset_files["event_cache"],
                [
                    ("t", "raw_events/t", np.float64),
                ],
            )
            self.preloaded_event["t"] = infer_event_timestamps_into_sec(self.preloaded_event["t"])
            self.opened_hdf5 = fileformat.open_hdf5(self.dataset_files["event_cache"])
            self._time_cache = self.preloaded_event["t"]
            self._len_cache = len(self._time_cache)
            self.set_image_cache()

    def load_event(self, start_index: int, end_index: int, cam: str = "left") -> np.ndarray:
        """Load events.
        The original hdf5 file contains (x, y, t, p),
        where x means in width direction, and y means in height direction. p is -1 or 1.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value, in sec. p is [-1, 1].
        """
        if self.event_file_format == "hdf5":
            return self.load_event_from_hdf5(start_index, end_index)
        return self.load_event_from_aedat(start_index, end_index)

    def load_event_from_hdf5(self, start_index: int, end_index: int, *args, **kwargs) -> np.ndarray:
        """Load events from Hdf5 cache file.
        The data format is space-separated .txt file.
        (timestamp x y polarity), but the x means in width direction, and y means in height direction.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value in second. p is [-1, +1].
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)
        if len(self) <= start_index:
            logger.error(f"Specified {start_index} to {end_index} index for {len(self)}.")
            raise IndexError
        events[:, 2] = np.array(self.preloaded_event["t"][start_index:end_index])
        events[:, 0] = np.array(
            self.opened_hdf5["raw_events/x"][start_index:end_index], dtype=np.int16
        )
        events[:, 1] = np.array(
            self.opened_hdf5["raw_events/y"][start_index:end_index], dtype=np.int16
        )
        events[:, 3] = np.array(self.opened_hdf5["raw_events/p"][start_index:end_index], dtype=bool)
        events[:, 3] = 2 * events[:, 3] - 1
        return events

    def load_event_from_aedat(
        self, start_index: int, end_index: int, *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    # Optical flow (GT)
    def load_calib(self, *args, **kwargs) -> dict:
        """Load calibration file.

        Outputs:
            (dict) ... {"K": camera_matrix, "D": distortion_coeff}
                camera_matrix (np.ndarray) ... [3 x 3] matrix.
                distortion_coeff (np.array) ... [5] array.
        """
        logger.warning("directly load calib_param is not implemented!! please use rectify instead.")
        return {}

    def get_calib_map(self, map_txt_x, map_txt_y):
        raise NotImplementedError

    def load_map_txt(self, map_txt):
        raise NotImplementedError
