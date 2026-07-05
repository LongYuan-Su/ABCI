# -*- coding: utf-8 -*-
# License: MIT License
"""
Start another process, define a framework for offline modeling and online processing with three functions:
    pre(): for offline modeling;

    consume(): for online prediction;

    post(): for subsequent custom operations.

In the actual usage process, you only need to customize the operations of the above functions.
"""
from typing import List, Optional, Any
from abc import abstractmethod
import os
import multiprocessing
import queue

import numpy as np

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger  # type: ignore[no-redef]

logger = get_logger("worker")


class ProcessWorker(multiprocessing.Process):
    """Online processing.

    author: Lichao Xu

    Created on: 2021-04-01

    update log:
        2022-08-10 by Wei Zhao

    Parameters
    ----------
    timeout: float
        Timer setting.
    name: str
        Custom name for the online processing process.

    Attributes
    ----------
    daemon: bool
    _exit:
        Multiprocess event handling.
    _in_queue: queue
        Data sharing between the online processing process and the main process.

    Tip
    ----
    ..  code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: A example using brainflow. worker

        from brainflow. worker import ProcessWorker
        class FeedbackWorker(ProcessWorker):
            def __init__():
                #Initialization

            def pre(self):
                #Off-line modeling

                #Online processing of data flow between stimulus interfaces
                info = StreamInfo(
                    name='meta_feedback',
                    type='Markers',
                    channel_count=1,
                    nominal_srate=0,
                    channel_format='int32',
                    source_id=self.lsl_source_id)
                self.outlet = StreamOutlet(info)
                print('waiting connection...')
                while not self._exit:
                    if self.outlet.wait_for_consumers(1e-3):
                        break
                print('Connected')

            def consume(self, data) :
                #Online processing
                if self.outlet.have_consumers ():
                    self.outlet.push_sample(“online results，list")

            def post(self):
                pass

    """

    def __init__(self, timeout: float = 1e-3, name: Optional[str] = None):
        multiprocessing.Process.__init__(self)
        self.daemon = False
        self._exit = multiprocessing.Event()
        self._in_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.timeout = timeout
        self.worker_name = name

    def put(self, data):
        """Put the data in the queue

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Parameters
        ----------
        data: ndarray, shape(n_samples, n_channels+1)
            Single trial of online data.

        """

        logger.info(
            "put samples in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._in_queue.put(data)

    def run(self):
        """
        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Online processing process:
            ① Customize the `pre()` function to build a model using offline data.

            ② Clear the queue and wait for data retrieval thread in the main process to get data within a fixed time.

            ③ Customize the `consume()` function to process online data and provide feedback.

            ④ Customize the `post()` function to perform subsequent operations.

            ⑤ Wait for the next online label to start the next online processing.

            ⑥ Close the online processing process, clear the queue, and stop online experiments.

        """
        logger.info(
            "start worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.clear()
        logger.info(
            "pre hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.pre()
        self.clear_queue()
        while not self._exit.is_set():
            try:
                data = self._in_queue.get(timeout=self.timeout)
                logger.info(
                    "consume samples in worker-{}".format(
                        self.worker_name if self.worker_name else os.getpid()
                    )
                )
                self.consume(data)
            except queue.Empty:
                # if queue is empty, loop to wait for next data until exiting
                pass
        logger.info(
            "post hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.post()
        self.clear_queue()
        logger.info(
            "worker{} exit".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )

    @abstractmethod
    def pre(self):
        """Custom function to build a model using offline data.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        pass

    @abstractmethod
    def consume(self, data):
        """Custom function to process online data.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Parameters
        ----------
        data: ndarray, shape(n_samples, n_channels+1)
            Single trial of online data.

        """
        pass

    @abstractmethod
    def post(self):
        pass

    def stop(self):
        """Stop the online processing process.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        logger.info(
            "stop worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.set()

    def settimeout(self, timeout=0.01):
        """Set the timer.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        self.timeout = timeout

    def clear_queue(self):
        """Clear the queue.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        logger.info(
            "clearing queue items in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        while True:
            try:
                self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                break
        logger.info(
            "all queue items in worker-{} are cleared".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )


class ContinuousWorker(multiprocessing.Process):
    """Continuous online processing worker (no trigger required).

    Suitable for devices without trigger channels, such as OpenBCI WiFi
    Shield and raw LSL streams.  Runs in a separate process with a
    sliding-window lifecycle: ``pre()`` → periodic ``maintain()`` → ``post()``.

    An optional *feat_queue* sends feature dicts back to the main thread
    for GUI visualisation.

    Parameters
    ----------
    timeout : float
        Queue polling interval in seconds.
    name : str, optional
        Custom worker name for logging.
    window_samples : int
        Number of samples passed to each ``maintain()`` call.
    step_samples : int
        How many new samples must accumulate before triggering ``maintain()``.
    """

    def __init__(
        self,
        timeout: float = 1e-3,
        name: Optional[str] = None,
        window_samples: int = 500,
        step_samples: int = 125,
    ):
        multiprocessing.Process.__init__(self)
        self.daemon = False
        self._exit = multiprocessing.Event()
        self._in_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self._feat_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.timeout = timeout
        self.worker_name = name
        self.window_samples = int(window_samples)
        self.step_samples = int(step_samples)

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def put(self, data):
        """Push a data chunk into the worker queue.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_channels, n_new)`` — raw or filtered data.
        """
        logger.info(
            "put samples in continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._in_queue.put(data)

    def get_features(self) -> Optional[Any]:
        """Non-blocking poll of the feature queue.

        Returns
        -------
        dict or None
            Most recent feature dict from ``maintain()``, or *None*.
        """
        try:
            return self._feat_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Signal the worker to exit."""
        logger.info(
            "stop continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.set()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """Orchestrate the continuous processing cycle."""
        logger.info(
            "start continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.clear()
        logger.info(
            "pre hook executed in continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.pre()
        self.clear_queue()

        # Sliding-window accumulator (list of arrays → concatenated)
        accumulator: List[np.ndarray] = []
        accumulated_samples: int = 0

        while not self._exit.is_set():
            try:
                data = self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                continue

            # data is expected to be (n_ch, n_new)
            if data is None or (hasattr(data, "size") and data.size == 0):
                continue

            accumulator.append(data)
            accumulated_samples += data.shape[1]

            if accumulated_samples >= self.step_samples:
                # Build a contiguous window
                window = np.concatenate(accumulator, axis=1)
                # Keep only the most recent *window_samples*
                if window.shape[1] > self.window_samples:
                    window = window[:, -self.window_samples:]
                logger.info(
                    "maintain called in continuous-worker-{}".format(
                        self.worker_name if self.worker_name else os.getpid()
                    )
                )
                result = self.maintain(window)
                if result is not None:
                    self._feat_queue.put(result)
                # Reset accumulator — keep last window_samples for overlap
                accumulator = [window]
                accumulated_samples = window.shape[1]

        logger.info(
            "post hook executed in continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.post()
        self.clear_queue()
        self._feat_queue.put(None)  # sentinel to notify GUI we stopped
        logger.info(
            "continuous-worker-{} exit".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def pre(self):
        """Override: one-time initialisation (load models, create outlets).

        Called once when the worker process starts.
        """
        pass

    def maintain(self, data):
        """Override: periodic processing of a sliding data window.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_channels, window_samples)``.

        Returns
        -------
        dict or None
            Optional feature dict that will be forwarded to
            ``get_features()`` in the main thread.
        """
        pass

    def post(self):
        """Override: cleanup (save results, close connections).

        Called once when the worker process exits.
        """
        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def settimeout(self, timeout: float = 0.01):
        """Set the queue polling interval."""
        self.timeout = timeout

    def clear_queue(self):
        """Drain the input queue."""
        logger.info(
            "clearing queue items in continuous-worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        while True:
            try:
                self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                break
        logger.info(
            "all queue items in continuous-worker-{} are cleared".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
