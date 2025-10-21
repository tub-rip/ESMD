import logging
import os, glob

import numpy as np

from ..utils import check_file_utils, swap_event
from ..utils import infer_event_timestamps_into_sec
from . import DataLoaderBase, fileformat, FileTypeNotSupportedError

logger = logging.getLogger(__name__)


class Dnd21DataLoader(DataLoaderBase):
    """Dataloader class for DND21 Dataset
    https://sites.google.com/view/dnd21/datasets?authuser=0
    """

    NAME = "DND21"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.gt_flow_available = False  # always false
        self._make_hdf5_cache = True
        self.opened_noise = None

    def post_set_sequence(self):
        if self.event_file_format == "hdf5":
            logger.info("Optimize data loading speed. Loading events...")
            # Load only t
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

        self.trigger_available = self.gt_depth_available = self.gt_pose_available = False
        self.frame_available = self.frame_try_to_load

    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `outdoot_day2`.

        Returns:
            sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        # data_path = os.path.join(self.dataset_dir, sequence_name)
        data_path: str = os.path.join(self.dataset_dir, sequence_name)
        print(data_path)
        event_cache_file = os.path.join(data_path, "events.hdf5")  # hdf5 file support
        aedat_list = glob.glob(os.path.join(data_path, "*.aedat"))
        if len(aedat_list) > 0:
            event_aedat_file = aedat_list[0]
        else:
            event_aedat_file = None
        if check_file_utils(event_cache_file):
            self.event_file_format = "hdf5"
        elif check_file_utils(event_aedat_file):
            self.event_file_format = "aedat"
        # Cache
        if self.event_file_format != "hdf5" and self._make_hdf5_cache:
            if self.event_file_format == "aedat":
                ev_iter = fileformat.IteratorAedatEvent(event_aedat_file)
            else:
                raise FileTypeNotSupportedError(f"{self.event_file_format} is not supported.")
            data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}  # type: ignore
            fileformat.convert_iterator_access_to_hdf5(ev_iter, event_cache_file, data_keys)
            self.event_file_format = "hdf5"
        logger.info(f"Event data format: {self.event_file_format}")

        # # frames
        # frame_cache_file = os.path.join(data_path, "frames.hdf5")  # hdf5 file support
        # if check_file_utils(frame_cache_file):
        #     self.frame_file_format = "hdf5"
        # elif check_file_utils(event_aedat_file):
        #     self.frame_file_format = "aedat"  # frame is encoded with event files..
        # # Cache
        # if self.frame_file_format != "hdf5" and self._make_hdf5_cache:
        #     fr_iter = fileformat.IteratorAedat4Frame(event_aedat_file)
        #     data_keys = {"frame": "frames/raw", "t": "frames/t"}  # type: ignore
        #     fileformat.convert_iterator_access_to_hdf5(
        #         fr_iter, frame_cache_file, data_keys, image_keys=["frame"]
        #     )
        #     self.frame_file_format = "hdf5"
        logger.info(f"Frame data format: {self.frame_file_format}")
        sequence_file = {
            "event_dat": event_aedat_file,  # dat
            "event_cache": event_cache_file,  # hdf5
            # "frame_cache": frame_cache_file,  # hdf5
        }
        return sequence_file

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
        raise FileTypeNotSupportedError(f"File format {self.event_file_format} not supported for event data.")

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
        events = swap_event(events, 0, self._HEIGHT - 1)
        if self.auto_undistort:
            events = self.undistort(events)
        return events

    # Optical flow (GT)
    def gt_time_list(self):
        raise NotImplementedError

    def eval_frame_time_list(self):
        raise NotImplementedError

    def get_gt_time(self, index: int) -> list:
        raise NotImplementedError

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

    def setup_noise(self, noise_type: str):
        noise_file_pair = {
            'background_activity_light': 'Davis346blue-2020-10-17T13-42-27+0200-00000002-0 light background activity.aedat',
            'background_activity_dark': 'Davis346blue-2020-10-19T19-50-42+0200-00000002-0 dark BA activity noise.aedat',
            'dark': 'Davis346blue-2021-06-14T20-09-08+0200-00000003-0 noise dark 5p3Hz.aedat',
            'leak_noise': 'Davis346blue-2021-06-20T09-23-20+0200-00000003-0 50m leak noise light.aedat',
        }
        noise_aedat_file = os.path.join(self.dataset_dir, 'measured-noise',  noise_file_pair[noise_type])
        noise_cache_file = noise_aedat_file.replace('.aedat', '.hdf5')
        if not check_file_utils(noise_cache_file):
            noise_iter = fileformat.IteratorAedatEvent(noise_aedat_file)
            data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}  # type: ignore
            fileformat.convert_iterator_access_to_hdf5(noise_iter, noise_cache_file, data_keys)
        self.noise_file = noise_cache_file
        self.opened_noise = fileformat.open_hdf5(self.noise_file)

    def mix_noise(self, signal_event: np.ndarray, noise_num: int):
        """
        Args:
            signal_event (np.ndarray): _description_
            noise_num (int): _description_

        Returns:
            _type_: signal+noise events and index. Index 1 is signal and 0 is noise.
        """
        max_n_events = len(self.opened_noise['raw_events/x'])
        if noise_num > max_n_events:
            raise ValueError(f"Exceeds max num of noise {max_n_events = }")
        
        start_index = np.random.randint(max_n_events - noise_num)  # start index
        end_index = start_index + noise_num

        noise_events = np.zeros((noise_num, 4), dtype=np.float64)
        noise_events[:, 2] = infer_event_timestamps_into_sec(np.array(self.opened_noise["raw_events/t"][start_index:end_index]))
        noise_events[:, 0] = np.clip(
            np.array(self.opened_noise["raw_events/x"][start_index:end_index], dtype=np.int16),
            0, self._HEIGHT)
        noise_events[:, 1] = np.clip(
            np.array(self.opened_noise["raw_events/y"][start_index:end_index], dtype=np.int16),
            0, self._WIDTH
        )
        noise_events[:, 3] = np.array(self.opened_noise["raw_events/p"][start_index:end_index], dtype=bool)
        noise_events[:, 3] = 2 * noise_events[:, 3] - 1

        signal_t_min = signal_event[:, 2].min()
        signal_t_max = signal_event[:, 2].max()
        noise_t_min = noise_events[:, 2].min()
        noise_t_max = noise_events[:, 2].max()
        # print('fffffffffff')
        # print(f'{noise_events.max(axis=0)}')
        # print(f'{noise_events.min(axis=0)}')

        noise_events[:, 2] = (noise_events[:, 2] - noise_t_min) / (noise_t_max - noise_t_min)
        noise_events[:, 2] = noise_events[:, 2] * (signal_t_max - signal_t_min) +  signal_t_min
        
        sn_array = np.zeros(len(signal_event) + len(noise_events))
        sn_array[:len(signal_event)] = 1   # 1 is signal, 0 is noise
        sn_event = np.concatenate([signal_event, noise_events], axis=0)        
        indices = sn_event[:, 2].argsort()
        return sn_event[indices], sn_array[indices]

