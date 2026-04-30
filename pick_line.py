import cv2
import numpy as np

VIDEO_PATH = "CRK01.mp4"
TARGET_W, TARGET_H = 1920, 1080
DISPLAY_W, DISPLAY_H = 1280, 720

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read video.")
    exit()

original = cv2.resize(frame, (TARGET_W, TARGET_H))
display = cv2.resize(original, (DISPLAY_W, DISPLAY_H))
points_display = []
points_orig = []

def draw(img, cursor=None):
    out = img.copy()
    if cursor:
        x, y = cursor
        x_orig = int(x * TARGET_W / DISPLAY_W)
        y_orig = int(y * TARGET_H / DISPLAY_H)
        cv2.line(out, (x, 0), (x, DISPLAY_H), (255, 255, 0), 1)
        cv2.line(out, (0, y), (DISPLAY_W, y), (255, 255, 0), 1)
        cv2.putText(out, f"({x_orig}, {y_orig})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for p in points_display:
        cv2.circle(out, p, 6, (0, 0, 255), -1)
    if len(points_display) == 2:
        cv2.line(out, points_display[0], points_display[1], (0, 255, 0), 4)
        mid = ((points_display[0][0] + points_display[1][0]) // 2,
               (points_display[0][1] + points_display[1][1]) // 2)
        cv2.putText(out, "COUNTING LINE", (mid[0] - 60, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Pick Line - Click 2 points", out)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        draw(display, cursor=(x, y))
    elif event == cv2.EVENT_LBUTTONDOWN:
        x_orig = int(x * TARGET_W / DISPLAY_W)
        y_orig = int(y * TARGET_H / DISPLAY_H)
        points_display.append((x, y))
        points_orig.append((x_orig, y_orig))
        print(f"Point {len(points_orig)}: display=({x}, {y}) -> 1080p=({x_orig}, {y_orig})")
        if len(points_orig) == 2:
            print(f"\nLINE_START = Point({points_orig[0][0]}, {points_orig[0][1]})")
            print(f"LINE_END   = Point({points_orig[1][0]}, {points_orig[1][1]})")
        draw(display, cursor=(x, y))

draw(display)
cv2.setMouseCallback("Pick Line - Click 2 points", on_mouse)
print("Move mouse to see coordinates. Click 2 points to define your counting line.")
print("Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
