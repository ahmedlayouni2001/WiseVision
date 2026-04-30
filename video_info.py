import cv2
video_path ="CRK01.mp4" 

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frames / fps if fps > 0 else 0
cap.release()

print(f"Resolution : {width}x{height}")
print(f"FPS        : {fps}")
print(f"Frames     : {frames}")
print(f"Duration   : {duration:.1f}s ({duration/60:.1f}min)")
