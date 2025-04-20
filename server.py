from flask import Flask, Response, jsonify, request, send_file, render_template_string, url_for, make_response, abort, render_template
from ultralytics import YOLO
import cv2
import os
import uuid
from flask_cors import CORS
import re
import subprocess

app = Flask(__name__)
CORS(app)

# Load your locally trained Roboflow YOLOv8 model
model = YOLO("weights.pt")  # Replace with your actual model path if different

# Set up folders
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(1)

current_state = "Unknown"  # Tracks last classification result

def classify_prediction(results):
    global current_state
    unsafe_keywords = ["unsafe", "on stomach", "face down", "suffocating"]  # Adjust to match your actual class names
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])].lower()
            if any(unsafe in label for unsafe in unsafe_keywords):
                current_state = "Unsafe"
                return "Unsafe"
    current_state = "Safe"
    return "Safe"

def generate_frames():
    global current_state
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        classify_prediction(results)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = model.names[int(box.cls[0])]
                confidence = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"classification": current_state})

@app.route('/')
def index():
    return "<h1>Noctra Stream</h1><img src='/video_feed' width='100%'>"

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No video part", 400

    file = request.files["video"]
    if file.filename == "":
        return "No selected file", 400

    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"{video_id}_annotated.mp4")
    file.save(video_path)
    print(f"Saved video to {video_path}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = model.names[int(box.cls[0])]
                confidence = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(annotated_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    # Re-encode the annotated video to H.264 using ffmpeg
    h264_path = annotated_path.replace(".mp4", "_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", annotated_path,
        "-vcodec", "libx264", "-crf", "23", h264_path
    ])


    annotated_url = url_for('serve_annotated_video', filename=f"{video_id}_annotated_h264.mp4", _external=True)
    print(annotated_url)
    return jsonify({ "annotated_url": annotated_url })

@app.route("/annotated/<filename>")
def serve_annotated_video(filename):
    path = os.path.join(ANNOTATED_FOLDER, filename)

    if not os.path.exists(path):
        abort(404)

    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(path, mimetype='video/mp4')

    size = os.path.getsize(path)
    byte1, byte2 = 0, None

    # Parse the range header manually
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if m:
        byte1 = int(m.group(1))
        if m.group(2):
            byte2 = int(m.group(2))

    byte2 = byte2 or size - 1
    length = byte2 - byte1 + 1

    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    rv.headers.add("Content-Disposition", f"inline; filename={filename}")
    return rv

@app.route("/test", methods=["GET", "POST"])
def test_page():
    video_url = None

    if request.method == "POST":
        if "video" not in request.files or request.files["video"].filename == "":
            return render_template("test.html", video_url=None)

        file = request.files["video"]
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
        annotated_path = os.path.join(ANNOTATED_FOLDER, f"{video_id}_annotated.mp4")
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = model.names[int(box.cls[0])]
                    confidence = box.conf[0].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(annotated_path, fourcc, 20.0, (width, height))

            out.write(frame)

        cap.release()
        if out:
            out.release()

        video_url = f"/annotated/{video_id}_annotated.mp4"

    return render_template("test.html", video_url=video_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
