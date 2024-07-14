import logging
import os
import queue
import subprocess
import threading


class VideoStreamer:
    def __init__(self, output_dir, frame_rate, width, height):
        self.output_dir = os.path.abspath(output_dir)
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
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
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            print(f"Warning: Frame queue is full. Current size: {self.frame_queue.qsize()}")

    def _encode_stream(self):
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.frame_rate),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-f",
            "hls",
            "-hls_time",
            "2",
            "-hls_list_size",
            "5",
            "-hls_flags",
            "delete_segments",
            os.path.join(self.output_dir, "stream.m3u8"),
        ]

        self.logger.info(f"Starting FFmpeg process with command: {' '.join(command)}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        if process.stdin is None:
            self.logger.error("Error: Unable to open stdin for FFmpeg process")
            return

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                process.stdin.write(frame.tobytes())
            except queue.Empty:
                continue

        process.stdin.close()
        process.wait()

        # Log FFmpeg output
        output, error = process.communicate()
        if error:
            self.logger.error(f"FFmpeg error: {error.decode()}")
        self.logger.info("FFmpeg process finished")
