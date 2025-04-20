from flask import Flask, Response, jsonify, request, send_file, url_for, abort
from ultralytics import YOLO
import cv2
import os
import uuid
from flask_cors import CORS
import re
import subprocess
import threading
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import utils

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("weights.pt")

# Setup folders
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Webcam
cap = cv2.VideoCapture(1)

# Current video classification state
detection_state = "Unknown"

# EEG globals
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

eeg_metrics = {
    "alpha-delta": None,
    "beta-theta": None,
    "theta-alpha": None,
    "alpha-delta-result": None,
    "beta-theta-result": None,
    "theta-alpha-result": None,
    "status": None
}

def eeg_thread():
    print('🔍 Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        print('❌ No EEG stream found.')
        return

    inlet = StreamInlet(streams[0], max_chunklen=12)
    fs = int(inlet.info().nominal_srate())

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    band_buffer = np.zeros((int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1)), 4))
    filter_state = None

    print('✅ EEG stream acquired. Starting processing...')
    while True:
        eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        if len(eeg_data) == 0:
            continue

        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
        eeg_buffer, filter_state = utils.update_buffer(eeg_buffer, ch_data, notch=True, filter_state=filter_state)
        data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
        band_powers = utils.compute_band_powers(data_epoch, fs)
        band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
        smooth_band_powers = np.mean(band_buffer, axis=0)

        alpha = smooth_band_powers[Band.Alpha] / smooth_band_powers[Band.Delta]
        beta = smooth_band_powers[Band.Beta] / smooth_band_powers[Band.Theta]
        theta = smooth_band_powers[Band.Theta] / smooth_band_powers[Band.Alpha]

        eeg_metrics["alpha-delta"] = float(alpha)
        eeg_metrics["beta-theta"] = float(beta)
        eeg_metrics["theta-alpha"] = float(theta)

        if alpha < 0.8:
            eeg_metrics["alpha-delta-result"] = "possible abnormal suppression or deep unconscious state"
        elif alpha < 1.5:
            eeg_metrics["alpha-delta-result"] = "possibly deep sleep or relaxed state"
        else:
            eeg_metrics["alpha-delta-result"] = "relaxed wakefulness or healthy sleep cycle"
        
        if beta < 0.6:
            eeg_metrics["beta-theta-result"] = "possible cortical underactivity"
        elif beta < 1.0:
            eeg_metrics["beta-theta-result"] = "may be asleep or drowsy"
        else:
            eeg_metrics["beta-theta-result"] = "healthy, cognitively alert (unusual at night unless waking)"

        if theta > 2.5:
            eeg_metrics["theta-alpha-result"] = "potential abnormal drowsiness or suppression"
        elif theta > 1.5:
            eeg_metrics["theta-alpha-result"] = "deep sleep or transitional state"
        else:
            eeg_metrics["theta-alpha-result"] = "healthy, calm wakefulness or light sleep"
        
        

def classify_prediction(results):
    global detection_state
    unsafe_keywords = ["unsafe", "safe"]
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])].lower()
            if any(unsafe in label for unsafe in unsafe_keywords):
                detection_state = "Unsafe"
                return "Unsafe"
    detection_state = "Safe"
    return "Safe"

def generate_frames():
    global detection_state
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
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "classification": detection_state,
        "eeg": eeg_metrics["status"],
        "eeg_values": {
            "alpha-delta": eeg_metrics["alpha-delta"],
            "beta-theta": eeg_metrics["beta-theta"],
            "theta-alpha": eeg_metrics["theta-alpha"],
            "alpha-delta-result": eeg_metrics["alpha-delta-result"],
            "beta-theta-result": eeg_metrics["beta-theta-result"],
            "theta-alpha-result": eeg_metrics["theta-alpha-result"]
        }
    })

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
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(annotated_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    # Re-encode using ffmpeg
    h264_path = annotated_path.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", annotated_path, "-vcodec", "libx264", "-crf", "23", h264_path])

    annotated_url = url_for('serve_annotated_video', filename=f"{video_id}_annotated_h264.mp4", _external=True)
    return jsonify({"annotated_url": annotated_url})

@app.route("/annotated/<filename>")
def serve_annotated_video(filename):
    path = os.path.join(ANNOTATED_FOLDER, filename)
    if not os.path.exists(path):
        abort(404)

    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(path, mimetype='video/mp4')

    size = os.path.getsize(path)
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    byte1, byte2 = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
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

if __name__ == '__main__':
    threading.Thread(target=eeg_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
