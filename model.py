import torch
from torchvision.models import resnet50, ResNet50_Weights 
import os
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to read frames from a video
def read_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        frames.append(frame)
    cap.release()
    return frames

# # Function to detect faces in frames and save them
# def detect_and_save_faces(video_path, output_dir):
#     frames = read_frames(video_path)
#     for i, frame in enumerate(frames):
#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
#         # If faces are detected, save them
#         if len(faces) > 0:
#             for j, (x, y, w, h) in enumerate(faces):
#                 face = frame[y:y+h, x:x+w]
#                 save_path = os.path.join(output_dir, f'video_{i}_face_{j}.jpg')
#                 cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

video_dir = '/home/stamatis/3rd bio/videos'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
output_dir = '/home/stamatis/3rd bio/out_videos'
output_dir2 = '/home/stamatis/3rd bio/out_torch_videos'
output_dir3 = '/home/stamatis/3rd bio/output_frames'



# for video_file in video_files:
#     video_path = os.path.join(video_dir, video_file)
#     output_video_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
#     os.makedirs(output_video_dir, exist_ok=True)
#     detect_and_save_faces(video_path, output_video_dir)




weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
model = torch.nn.Sequential(*list(model.children())[:-1])
#batch = preprocess(output_dir3).unsqueeze(0)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to save frames from videos
def preprocess_and_extract_features(frames):
    features = []
    for frame in frames:
        # Convert frame to PIL image
        frame_pil = Image.fromarray(frame)
        # Preprocess frame
        input_tensor = preprocess(frame_pil)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        # Extract features
        with torch.no_grad():
            output = model(input_batch)
        features.append(output.squeeze().numpy())
    return features

# for video_file in os.listdir(video_dir):
#     if video_file.endswith('.mp4'):
#         video_path = os.path.join(video_dir, video_file)
#         frames = read_frames(video_path)
#         feature_vectors = preprocess_and_extract_features(frames)

# Function to save feature vectors as CSV files
def save_feature_vectors(feature_vectors, output_dir, video_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, vector in enumerate(feature_vectors):
        file_name = os.path.join(output_dir, f"{video_name}_frame_{i}.csv")
        np.savetxt(file_name, vector, delimiter=",")

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames = read_frames(video_path)
        feature_vectors = preprocess_and_extract_features(frames)
        save_feature_vectors(feature_vectors, output_dir2, video_name)