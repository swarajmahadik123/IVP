from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for server-side plotting
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def detect_frame_changes(video_path, threshold=0.85, cooldown_frames=10):
    """Improved frame change detection with edge analysis and cooldown mechanism."""
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    changes = []
    differences = []
    cooldown = 0  # Cooldown counter to reduce redundant frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip processing during cooldown period
        if cooldown > 0:
            cooldown -= 1
            frame_count += 1
            continue

        # Preprocessing: Convert frame to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        if prev_frame is not None:
            # Calculate structural difference between consecutive frames
            structural_diff = cv2.absdiff(prev_frame, edges)
            diff_score = np.mean(structural_diff)

            # Calculate histogram difference between consecutive frames
            hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([edges], [0], None, [256], [0, 256])
            hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

            # Combined difference metric (structural + histogram difference)
            combined_diff = (1 - hist_diff) * 100 + diff_score
            differences.append(combined_diff)

            # Detect significant scene change based on threshold
            if combined_diff > threshold:
                frame_path = os.path.join(app.config['PROCESSED_FOLDER'], f'change_{frame_count}.jpg')
                cv2.imwrite(frame_path, frame)
                changes.append({
                    'frame': frame_count,
                    'image': f'processed/change_{frame_count}.jpg',
                    'diff': f"{combined_diff:.2f}"
                })
                # Activate cooldown period to skip redundant frames
                cooldown = cooldown_frames

        prev_frame = edges.copy()
        frame_count += 1

    cap.release()

    # Create difference plot for visualization
    plot_path = os.path.join(app.config['PROCESSED_FOLDER'], 'differences.png')
    plt.figure(figsize=(12, 4))
    plt.plot(differences)
    plt.title('Frame Difference Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference Score')
    plt.savefig(plot_path)
    plt.close()

    return changes, 'processed/differences.png'

@app.route('/', methods=['GET', 'POST'])
def index():
    """Home page for video upload."""
    if request.method == 'POST':
        file = request.files['video']
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            # Detect scene changes in the uploaded video
            changes, plot = detect_frame_changes(video_path)
            return render_template('results.html', changes=changes, plot=plot)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
