import argparse
import logging
from multiprocessing import Process
from multiprocessing import Queue

import frame_processor
import web_server
from shared import Config
from shared import shutdown_flag

logger = logging.getLogger(__name__)


def main(input_source: str, config: Config):
    # Create a shared queue for passing processed frames
    frame_queue = Queue()

    # Start the frame processor process
    logger.info("Starting frame processo...")
    frame_proc = Process(target=frame_processor.main, args=(input_source, frame_queue, config))
    frame_proc.start()
    logger.info("Frame processor process started.")

    # Start the web server process
    logger.info("Starting web server...")
    web_proc = Process(target=web_server.main, args=(frame_queue, config))
    web_proc.start()
    logger.info("Web server process started.")

    try:
        # Wait for the processes to finish
        frame_proc.join()
        web_proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    finally:
        # Set the shutdown flag to signal processes to stop
        shutdown_flag.set()

        # Wait for processes to finish
        frame_proc.join(timeout=5)
        web_proc.join(timeout=5)

        # If processes are still alive, terminate them
        if frame_proc.is_alive():
            frame_proc.terminate()
        if web_proc.is_alive():
            web_proc.terminate()

        logger.info("All processes have been shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    args = parser.parse_args()

    config = Config("./config.yaml")
    main(input_source=args.input_source, config=config)
