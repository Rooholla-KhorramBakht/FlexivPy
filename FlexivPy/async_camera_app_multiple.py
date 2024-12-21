import argparse
import time
from datetime import datetime
import cv2

from FlexivPy.vision import RealSenseCamera
from FlexivPy.robot.dds.flexiv_messages import EnvImage

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter

# Dictionary mapping camera IDs to their serial numbers
CAMERA_SERIAL_MAP = {
    0: "234222302193",
    1: "231622302407",
    2: "141222073965",  # example serial for camera ID 1
    # Add more IDs and serials as needed
}


class ImagePublisher:
    def __init__(self, camera_writers, dt=1.0 / 50.0, width=320, height=240):
        """
        camera_writers: a list of tuples (camera_id, realsense_camera, data_writer)
        """
        self.camera_writers = camera_writers
        self.dt = dt
        self.width = width
        self.height = height

    def run(self):
        while True:
            tic = time.time()
            # Process each camera in turn
            for cid, camera, writer in self.camera_writers:
                image = camera.color_frame
                if image is not None:
                    # Resize and convert image
                    image = cv2.resize(
                        image, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode(".jpg", image)
                    image_bytes = buffer.tobytes()

                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                    image_data = EnvImage(data=image_bytes, timestamp=timestamp)
                    writer.write(image_data)

            elapsed_time = time.time() - tic
            if elapsed_time > self.dt:
                print(
                    f"Warning: processing time {elapsed_time} exceeded desired time step {self.dt}."
                )
                # Sleep to maintain desired time step
            time.sleep(max(0, self.dt - elapsed_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-camera image publisher script.")
    parser.add_argument(
        "--camera_ids",
        nargs="+",
        type=int,
        default=[0],
        help="List of camera IDs to publish from (e.g. --camera_ids 0 1)",
    )
    parser.add_argument(
        "--topic_name",
        type=str,
        default="EnvImage_$camera_id",
        help="DDS topic name pattern",
    )
    parser.add_argument("--width", type=int, default=320, help="Output image width")
    parser.add_argument("--height", type=int, default=240, help="Output image height")
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 50.0,
        help="Desired time step between frames (in seconds)",
    )
    args = parser.parse_args()

    # Create a single DomainParticipant and Publisher for all cameras
    domain_participant = DomainParticipant(1)
    publisher = Publisher(domain_participant)

    camera_writers = []

    # Initialize all cameras and their corresponding DataWriters
    cameras = []
    for cid in args.camera_ids:
        camera_serial_no = CAMERA_SERIAL_MAP.get(cid, None)
        if camera_serial_no is None:
            raise ValueError(f"No serial number found for camera_id {cid}")

        # Create the camera
        camera = RealSenseCamera(
            camera_serial_no=camera_serial_no,
            VGA=True,
            enable_imu=False,
            enable_depth=False,
            enable_color=True,
            enable_ir=False,
            emitter_enabled=False,
        )
        cameras.append(camera)

        # Allow camera some time to initialize
        time.sleep(1)

        # Create a unique topic name for this camera
        topic_name = (
            args.topic_name.replace("$camera_id", str(cid))
            if "$camera_id" in args.topic_name
            else args.topic_name
        )
        topic = Topic(domain_participant, topic_name, EnvImage)
        data_writer = DataWriter(publisher, topic)

        # Store (cid, camera, data_writer) tuple
        camera_writers.append((cid, camera, data_writer))

    img_publisher = ImagePublisher(
        camera_writers=camera_writers, dt=args.dt, width=args.width, height=args.height
    )

    try:
        img_publisher.run()
    finally:
        # Clean up cameras on exit
        for camera in cameras:
            camera.close()
