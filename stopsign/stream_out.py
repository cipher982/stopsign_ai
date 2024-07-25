import logging
import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np


class VideoStreamer:
    def __init__(self, output_dir, frame_rate, width, height):
        self.output_dir = os.path.abspath(output_dir)
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.debug_mode = False
        self.debug_frame = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.encoding_thread = threading.Thread(target=self._encode_stream)
        self.encoding_thread.start()

    def stop(self):
        self.stop_event.set()
        self.encoding_thread.join()

    def add_frame(self, frame):
        if self.debug_mode:
            if self.debug_frame is None:
                self.debug_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    self.debug_frame, "Debug Static Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            frame = self.debug_frame

        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            self.logger.warning(f"Frame queue is full. Current size: {self.frame_queue.qsize()}")

    def _encode_stream(self):
        # fmt: off
        command = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.frame_rate),
            "-i", "-",
            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "30",  # GOP size
            "-b:v", "2M",  # Video bitrate
            "-f", "hls",
            "-hls_time", "4",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments+omit_endlist",
            "-hls_segment_type", "mpegts",
            os.path.join(self.output_dir, "stream.m3u8"),
        ]

        self.logger.info(f"Starting FFmpeg process with command: {' '.join(command)}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        if process.stdin is None:
            self.logger.error("Error: Unable to open stdin for FFmpeg process")
            return

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=1)
                    process.stdin.write(frame.tobytes())
                except queue.Empty:
                    continue
                except BrokenPipeError:
                    self.logger.error("BrokenPipeError: FFmpeg process may have terminated unexpectedly")
                    break
        except Exception as e:
            self.logger.error(f"Error in encoding stream: {str(e)}")
        finally:
            process.stdin.close()
            process.wait()

        # Log FFmpeg output
        output, error = process.communicate()
        if output:
            self.logger.info(f"FFmpeg output: {output.decode()}")
        if error:
            self.logger.error(f"FFmpeg error: {error.decode()}")

    def debug_static_image(self, duration_seconds=30):
        # Create a simple static image
        static_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(static_frame, "Debug Static Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            self.add_frame(static_frame)
            time.sleep(1 / self.frame_rate)

        self.logger.info("Finished sending static debug image")
