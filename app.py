import os
import sqlite3
from cs50 import SQL
import mimetypes
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import time
import threading


app = Flask(__name__, template_folder="template")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["OUTPUT_FOLDER"] = "static/outputs"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            # Save the file to disk
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            process_video()
            mimetype = mimetypes.guess_type(filename)[0]
            return render_template("video.html", filename=filename, mimetype=mimetype)
            # return "Processed and saved in output"
    else:
        return redirect(url_for("index"))


@app.route("/video")
def video():
    # Get the filename from the query parameter
    filename = request.args.get("filename")
    # Render the video page
    return render_template("video.html", filename=filename)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/image.jpg")
def image():
    return render_template("image.jpg")




@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/result")
def result():
    """Video streaming home page."""
    return render_template("result.html")


def calibrated_dist(p1, p2):
    return (
        (p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2
    ) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0


# code for live detection
# -------------------------------------------------------------------------------------------

@app.route("/gen_frames")
def gen_frames():
    # q = 0
    confid = 0.5
    threshold = 0.5
    labelsPath = "./coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(0)  # laptopcam
    # vs = cv2.VideoCapture(1)    #web cam
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    FR = 0
    writer = None
    (W, H) = (None, None)
    while True:
        success, frame = vs.read()
        if not success:
            break
        else:
            # frame = cv2.flip(frame, 1)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                q = W

            frame = frame[0:H, 200:q]
            (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if LABELS[classID] == "person":
                        if confidence > confid:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, threshold)

            if len(idxs) > 0:
                status = list()
                idf = idxs.flatten()
                close_pair = list()
                s_close_pair = list()
                center = list()
                dist = list()

                for i in idf:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    center.append([int(x + w / 2), int(y + h / 2)])
                    status.append(0)

                for i in range(len(center)):
                    for j in range(len(center)):
                        g = isclose(center[i], center[j])
                        if g == 1:
                            close_pair.append([center[i], center[j]])
                            status[i] = 1
                            status[j] = 1
                        elif g == 2:
                            s_close_pair.append([center[i], center[j]])
                            if status[i] != 1:
                                status[i] = 2
                            if status[j] != 1:
                                status[j] = 2

                total_p = len(center)
                low_risk_p = status.count(2)
                high_risk_p = status.count(1)
                safe_p = status.count(0)

                kk = 0

                for i in idf:
                    tot_str = "TOTAL COUNT: " + str(total_p)
                    high_str = "HIGH RISK COUNT: " + str(high_risk_p)
                    low_str = "LOW RISK COUNT: " + str(low_risk_p)
                    safe_str = "SAFE COUNT: " + str(safe_p)
                    sub_img = frame[H - 120 : H, 0:210]
                    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
                    res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)
                    frame[H - 120 : H, 0:210] = res
                    cv2.putText(frame,tot_str,(10, H - 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),1,)
                    cv2.putText(frame,safe_str,(10, H - 65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),1,)
                    cv2.putText(frame,low_str,(10, H - 40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 120, 255),1,)
                    cv2.putText(frame,high_str,(10, H - 15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 150),1,)

                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    if status[kk] == 1:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                    elif status[kk] == 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)
                    kk += 1

                    def insert_data():
                        db = sqlite3.connect("svda.db")
                        db.execute(
                            "INSERT INTO data (total_person, high_risk, low_risk, safe_person) VALUES (?, ?, ?, ?);",
                            (
                                total_p,
                                high_risk_p,
                                low_risk_p,
                                safe_p,
                            ),
                        )
                        db.commit()  # push to the database
                        db.close()  # close the connection

                    # Wait for a certain period of time (5 sec) and then execute the query
                    # timer = threading.Timer(5, insert_data)
                    # timer.start()

                for h in close_pair:
                    cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
                for b in s_close_pair:
                    cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
                cv2.waitKey(1)
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    vs.release()




# ------------------------------------------------------------------------------
# code for detection in video


@app.route("/process_video", methods=["GET", "POST"])
def process_video():
    confid = 0.5
    thresh = 0.3
    LABELS = open("./coco.names").read().strip().split("\n")
    np.random.seed(42)

    # Read the network
    net = cv2.dnn.readNetFromDarknet("./yolov3.cfg", "./yolov3.weights")

    # Get names of layers
    ln = net.getLayerNames()

    # Get indices of output layer
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    FR = 0
    # Get the path of the first file in the uploads folder
    upload_folder = app.config["UPLOAD_FOLDER"]
    files = os.listdir(upload_folder)
    if not files:
        return "No video file found in the upload folder"

    filename = files[0]
    video_path = os.path.join(upload_folder, filename)

    # Open the video file
    vs = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not vs.isOpened():
        # Try opening the video file with a different codec
        vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not vs.isOpened():
            # Failed to open video file with any codec
            return "Failed to open video file"

    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "output.mp4")

    # knowing whether file is opening or not
    print("opened video file")

    writer = None
    (W, H) = (None, None)

    fl = 0
    q = 0
    count = 0
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            q = W

        frame = frame[0:H, 200:q]
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if LABELS[classID] == "person":
                    if confidence > confid:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

        if len(idxs) > 0:
            idf = idxs.flatten()

            status = list()
            close_pair = list()
            s_close_pair = list()
            center = list()
            dist = list()
            for i in idf:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                center.append([int(x + w / 2), int(y + h / 2)])

                status.append(0)
            for i in range(len(center)):
                for j in range(len(center)):
                    g = isclose(center[i], center[j])

                    if g == 1:
                        close_pair.append([center[i], center[j]])
                        status[i] = 1
                        status[j] = 1
                    elif g == 2:
                        s_close_pair.append([center[i], center[j]])
                        if status[i] != 1:
                            status[i] = 2
                        if status[j] != 1:
                            status[j] = 2

            total_p = len(center)
            low_risk_p = status.count(2)
            high_risk_p = status.count(1)
            safe_p = status.count(0)

            kk = 0

            for i in idf:
                tot_str = "TOTAL COUNT: " + str(total_p)
                high_str = "HIGH RISK COUNT: " + str(high_risk_p)
                low_str = "LOW RISK COUNT: " + str(low_risk_p)
                safe_str = "SAFE COUNT: " + str(safe_p)
                sub_img = frame[H - 120 : H, 0:210]
                black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
                res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)
                frame[H - 120 : H, 0:210] = res
                cv2.putText(frame,tot_str,(10, H - 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),1,)
                cv2.putText(frame,safe_str,(10, H - 65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),1,)
                cv2.putText(frame,low_str,(10, H - 40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 120, 255),1,)
                cv2.putText(frame,high_str,(10, H - 15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 150),1,)

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if status[kk] == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

                elif status[kk] == 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 2)

                kk += 1
            for h in close_pair:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
            for b in s_close_pair:
                cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

        print("video being saved...", count)
        count += 1
        if writer is None:
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            writer = cv2.VideoWriter(
                output_path, fourcc, 12, (frame.shape[1], frame.shape[0]), True
            )

        writer.write(frame)
    writer.release()
    # print("Processing finished: open output.mp4")
    vs.release()
    time.sleep(3)
    print("Processing finished file saved in output")
    return "."


if __name__ == "__main__":
    app.run(debug=True)



# sqlite3 svda.db         database opening
# .mode csv                output will generate in csv
# .output file_name.csv    generate file
# SELECT * FROM table_name(data) ;       select whole table
# .quit          close sqlite3

# after generation of output.csv paste this on first line (Id,Total People,High Risk People,Low Risk People,Safe People,Timestamp)

# for performing sql queries first enter open database(step1) then run commands
