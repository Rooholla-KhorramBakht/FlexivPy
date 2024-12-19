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
    1: "231622302407",  # example serial number for camera ID 1
}


class ImagePublisher:
    def __init__(
        self,
        realsense_camera: RealSenseCamera,
        dt=0.01,
        topic_name="EnvImage",
        width=320,
        height=240,
    ):
        self.domain_participant = DomainParticipant()
        self.topic_state_image = Topic(self.domain_participant, topic_name, EnvImage)
        self.publisher_image = Publisher(self.domain_participant)
        self.writer_image = DataWriter(self.publisher_image, self.topic_state_image)
        self.dt = dt
        self.realsense_camera = realsense_camera
        self.width = width
        self.height = height

    def run(self):
        while True:
            tic = time.time()

            image = self.realsense_camera.color_frame

            if image is not None:
                image = cv2.resize(
                    image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )
                # Convert from RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".png", image)
                image_bytes = buffer.tobytes()

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                image_data = EnvImage(data=image_bytes, timestamp=timestamp)
                self.writer_image.write(image_data)

            elapsed_time = time.time() - tic
            time.sleep(max(0, self.dt - elapsed_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image publisher script.")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID")
    parser.add_argument(
        "--topic_name", type=str, default="EnvImage_$camera_id", help="DDS topic name"
    )
    parser.add_argument("--width", type=int, default=320, help="Output image width")
    parser.add_argument("--height", type=int, default=240, help="Output image height")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Desired time step between frames (in seconds)",
    )
    args = parser.parse_args()

    if args.topic_name.find("$camera_id") != -1:

        args.topic_name = args.topic_name.replace("$camera_id", str(args.camera_id))

    # Lookup the camera serial number based on camera_id
    camera_serial_no = CAMERA_SERIAL_MAP.get(args.camera_id, None)
    if camera_serial_no is None:
        raise ValueError(f"No serial number found for camera_id {args.camera_id}")

    camera = RealSenseCamera(VGA=True, camera_serial_no=camera_serial_no)

    time.sleep(1)

    img_publisher = ImagePublisher(
        realsense_camera=camera,
        dt=args.dt,
        topic_name=args.topic_name,
        width=args.width,
        height=args.height,
    )

    try:
        img_publisher.run()
    finally:
        camera.close()
