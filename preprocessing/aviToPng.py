import cv2
import os

### SET THESE VARIABLES ###
avi_folder = "/media/nick/C8EB-647B/Data/2_chs/2025-03-04/Exp_4"
file_names = ["C", "M"]
frames_directory_base = "/media/nick/C8EB-647B/Data/processed/2_chs/frames/Exp_4"
###########################

def count_frames(file_name):
    file_path = os.path.join(avi_folder, file_name + ".avi")
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Function to process a single video file
def process_video(file_name):
    frames_directory = os.path.join(frames_directory_base, file_name)
    os.makedirs(frames_directory, exist_ok=True)
    file_path = os.path.join(avi_folder, file_name + ".avi")

    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    batch_size = 100  # Number of frames to process in each batch
    frame_batch = []

    print("Starting to write frames for:", file_name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{frames_directory}/frame_{frame_count:05d}.png"
        frame_batch.append((frame, frame_filename))
        frame_count += 1

        # Write frames in batch
        if len(frame_batch) >= batch_size:
            for frame, frame_filename in frame_batch:
                cv2.imwrite(frame_filename, frame)
            frame_batch = []

        if frame_count % 1000 == 0:
            print(f"{frame_count} frames saved for {file_name}")

    # Write any remaining frames in the batch
    for frame, frame_filename in frame_batch:
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"Finished writing frames for {file_name}")

# Estimate total frames
total_estimated_frames = sum(count_frames(file) for file in file_names)
print(f"Estimated total frames to process: {total_estimated_frames}")

# Process each file
for file_name in file_names:
    process_video(file_name)
