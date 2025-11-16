#!/usr/bin/env python
"""
Live video face recognition with CLI options.
"""
import os
import cv2
import imutils
import time
import pickle
import numpy as np
import argparse
from imutils.video import FPS
from imutils.video import VideoStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", default="0",
                    help="camera source (index) or path to video file")
    ap.add_argument("-o", "--output", required=False,
                    help="optional path to output video file (saved annotated video)")
    ap.add_argument("--no-display", action="store_true",
                    help="don't show display window")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("--snapshot", required=False,
                    help="optional path to save a single annotated frame and exit")
    args = vars(ap.parse_args())

    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer (embedder)...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    print("Loading classifier and label encoder...")
    if os.path.exists('output/recognizer'):
        recognizer = pickle.loads(open('output/recognizer', "rb").read())
    else:
        recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # initialize video source (camera index or video file)
    print("Starting video stream...")
    src = args.get("src")
    use_video_file = False
    try:
        src_int = int(src)
        vs = VideoStream(src=src_int).start()
        time.sleep(2.0)
    except Exception:
        # treat src as a video file path
        use_video_file = True
        vs = cv2.VideoCapture(src)

    # start the FPS throughput estimator
    fps = FPS().start()

    # prepare optional video writer
    writer = None
    if args.get("output"):
        out_path = args.get("output")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # loop over frames from the video file stream
    frame_count = 0
    while True:
        # grab the frame from the threaded video stream or video file
        if use_video_file:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
        else:
            frame = vs.read()

        # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # initialize writer if needed (once we know frame size)
        if args.get("output") and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(args.get("output"), fourcc, 20.0, (w, h))

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args.get("confidence"):
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        # write frame to output video if requested
        if writer is not None:
            writer.write(frame)

        # optionally save a single snapshot and exit
        if args.get("snapshot"):
            os.makedirs(os.path.dirname(args.get("snapshot")), exist_ok=True)
            cv2.imwrite(args.get("snapshot"), frame)
            print(f"Saved snapshot to {args.get('snapshot')}")
            break

        # show the output frame (unless disabled)
        if not args.get("no_display"):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        frame_count += 1

    # stop the timer and display FPS information
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if use_video_file:
        vs.release()
    else:
        vs.stop()


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Live video face recognition with CLI options.
"""
import os
import cv2
import imutils
import time
import pickle
import numpy as np
import argparse
from imutils.video import FPS
from imutils.video import VideoStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", default="0",
                    help="camera source (index) or path to video file")
    ap.add_argument("-o", "--output", required=False,
                    help="optional path to output video file (saved annotated video)")
    ap.add_argument("--no-display", action="store_true",
                    help="don't show display window")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("--snapshot", required=False,
                    help="optional path to save a single annotated frame and exit")
    args = vars(ap.parse_args())

    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer (embedder)...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    print("Loading classifier and label encoder...")
    if os.path.exists('output/recognizer'):
        recognizer = pickle.loads(open('output/recognizer', "rb").read())
    else:
        recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # initialize video source (camera index or video file)
    print("Starting video stream...")
    src = args.get("src")
    use_video_file = False
    try:
        src_int = int(src)
        vs = VideoStream(src=src_int).start()
        time.sleep(2.0)
    except Exception:
        # treat src as a video file path
        use_video_file = True
        vs = cv2.VideoCapture(src)

    # start the FPS throughput estimator
    fps = FPS().start()

    # prepare optional video writer
    writer = None
    if args.get("output"):
        out_path = args.get("output")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # loop over frames from the video file stream
    frame_count = 0
    while True:
        # grab the frame from the threaded video stream or video file
        if use_video_file:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
        else:
            frame = vs.read()

        # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # initialize writer if needed (once we know frame size)
        if args.get("output") and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(args.get("output"), fourcc, 20.0, (w, h))

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args.get("confidence"):
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        # write frame to output video if requested
        if writer is not None:
            writer.write(frame)

        # optionally save a single snapshot and exit
        if args.get("snapshot"):
            os.makedirs(os.path.dirname(args.get("snapshot")), exist_ok=True)
            cv2.imwrite(args.get("snapshot"), frame)
            print(f"Saved snapshot to {args.get('snapshot')}")
            break

        # show the output frame (unless disabled)
        if not args.get("no_display"):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        frame_count += 1

    # stop the timer and display FPS information
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if use_video_file:
        vs.release()
    else:
        vs.stop()


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Live video face recognition with CLI options.
"""
import os
import cv2
import imutils
import time
import pickle
import numpy as np
import argparse
from imutils.video import FPS
from imutils.video import VideoStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", default="0",
                    help="camera source (index) or path to video file")
    ap.add_argument("-o", "--output", required=False,
                    help="optional path to output video file (saved annotated video)")
    ap.add_argument("--no-display", action="store_true",
                    help="don't show display window")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("--snapshot", required=False,
                    help="optional path to save a single annotated frame and exit")
    args = vars(ap.parse_args())

    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer (embedder)...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    print("Loading classifier and label encoder...")
    if os.path.exists('output/recognizer'):
        recognizer = pickle.loads(open('output/recognizer', "rb").read())
    else:
        recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # initialize video source (camera index or video file)
    print("Starting video stream...")
    src = args.get("src")
    use_video_file = False
    try:
        src_int = int(src)
        vs = VideoStream(src=src_int).start()
        time.sleep(2.0)
    except Exception:
        # treat src as a video file path
        use_video_file = True
        vs = cv2.VideoCapture(src)

    # start the FPS throughput estimator
    fps = FPS().start()

    # prepare optional video writer
    writer = None
    if args.get("output"):
        out_path = args.get("output")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # loop over frames from the video file stream
    frame_count = 0
    while True:
        # grab the frame from the threaded video stream or video file
        if use_video_file:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
        else:
            frame = vs.read()

        # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # initialize writer if needed (once we know frame size)
        if args.get("output") and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(args.get("output"), fourcc, 20.0, (w, h))

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args.get("confidence"):
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        # write frame to output video if requested
        if writer is not None:
            writer.write(frame)

        # optionally save a single snapshot and exit
        if args.get("snapshot"):
            os.makedirs(os.path.dirname(args.get("snapshot")), exist_ok=True)
            cv2.imwrite(args.get("snapshot"), frame)
            print(f"Saved snapshot to {args.get('snapshot')}")
            break

        # show the output frame (unless disabled)
        if not args.get("no_display"):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        frame_count += 1

    # stop the timer and display FPS information
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if use_video_file:
        vs.release()
    else:
        vs.stop()


if __name__ == '__main__':
    main()
 