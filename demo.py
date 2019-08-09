import argparse
import cv2
from butterfly import Butterfly

# Parse the arguments user entered.
parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--model", default="frozen_model.pb",
                    type=str, help="Frozen model file to import")
parser.add_argument("--list_ops", default="false",
                    type="bool", help="List all the ops in the model graph")
parser.add_argument("--image", default=None,
                    type=str, help="Input image file")
parser.add_argument("--video", default=None,
                    type=str, help="Input video file")
parser.add_argument("--cam", default=None,
                    type=int, help="Camera index number")
parser.add_argument("--input_node", default="",
                    type=str, help="The name of the input node")
parser.add_argument("--output_node", default="",
                    type=str, help="The name of the output node")
args = parser.parse_args()


if __name__ == '__main__':
    # Construct a Butterfly.
    fly = Butterfly(args.model, args.input_node, args.output_node)

    # Output all the ops name in the graph.
    if args.list_ops:
        for op in fly.list_ops():
            print(op)

    # Process an image.
    if args.image:
        image = cv2.imread(args.image)
        print(fly.run([image]))

    # Process video/cam.
    if args.cam is not None:
        video_source = args.cam
    elif args.video:
        video_source = args.video
    else:
        video_source = None

    if video_source is not None:
        cap = cv2.VideoCapture(video_source)
        while True:
            _, frame = cap.read()

            result = fly.run([frame])
            print(result)

            cv2.imshow("Preview", frame)
            if cv2.waitKey(27) == 27:
                break
