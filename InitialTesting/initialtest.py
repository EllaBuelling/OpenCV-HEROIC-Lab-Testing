import cv2 as cv
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "OpenCVTest.jpg")

# ---------------------------------------------------------------------------
# Settings (tune these for your lighting / camera)
# ---------------------------------------------------------------------------
# OpenCV HSV ranges: H = 0-179, S = 0-255, V = 0-255
# Menthol blue is a light, slightly greenish cyan-blue. These are a starting
# point; click on the image (see below) to sample real HSV values and adjust.
LOWER_BLUE = np.array([85, 40, 120])
UPPER_BLUE = np.array([110, 255, 255])

# Treat near-white pixels (low saturation, high brightness) as background so
# you can also get "% of the non-background area".
IGNORE_WHITE_BACKGROUND = True
WHITE_S_MAX = 25
WHITE_V_MIN = 200

MIN_BLOB_AREA_PX = 50      # ignore specks smaller than this
KERNEL_SIZE = 5            # morphological cleanup kernel
DISPLAY_MAX_WIDTH = 500    # per-panel width when showing results

# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------
img = cv.imread(img_path)
if img is None:
    sys.exit(f"Could not read the image at: {img_path}")

blurred = cv.GaussianBlur(img, (5, 5), 0)
hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

# ---------------------------------------------------------------------------
# Build the blue mask
# ---------------------------------------------------------------------------
mask = cv.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove specks
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes

# Remove blobs below the minimum area
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
clean_mask = np.zeros_like(mask)
kept = [c for c in contours if cv.contourArea(c) >= MIN_BLOB_AREA_PX]
cv.drawContours(clean_mask, kept, -1, 255, thickness=cv.FILLED)
mask = clean_mask

# ---------------------------------------------------------------------------
# Compute statistics
# ---------------------------------------------------------------------------
total_px = mask.size
blue_px = int(cv.countNonZero(mask))
pct_total = 100.0 * blue_px / total_px

print("=" * 50)
print(f"Image:               {os.path.basename(img_path)}")
print(f"Dimensions:          {img.shape[1]} x {img.shape[0]} px")
print(f"Total pixels:        {total_px}")
print(f"Menthol blue pixels: {blue_px}")
print(f"% of whole image:    {pct_total:.2f}%")

if IGNORE_WHITE_BACKGROUND:
    white_mask = cv.inRange(
        hsv, np.array([0, 0, WHITE_V_MIN]), np.array([179, WHITE_S_MAX, 255])
    )
    foreground_px = total_px - int(cv.countNonZero(white_mask))
    if foreground_px > 0:
        pct_fg = 100.0 * blue_px / foreground_px
        print(f"Non-white pixels:    {foreground_px}")
        print(f"% of non-white area: {pct_fg:.2f}%")

if blue_px > 0:
    mean_bgr = cv.mean(img, mask=mask)[:3]
    mean_hsv = cv.mean(hsv, mask=mask)[:3]
    print(f"Number of blobs:     {len(kept)}")
    print(f"Largest blob:        {int(max(cv.contourArea(c) for c in kept))} px")
    print(f"Mean color (RGB):    ({mean_bgr[2]:.0f}, {mean_bgr[1]:.0f}, {mean_bgr[0]:.0f})")
    print(f"Mean color (HSV):    ({mean_hsv[0]:.0f}, {mean_hsv[1]:.0f}, {mean_hsv[2]:.0f})")
else:
    print("No menthol blue detected with the current HSV range.")
print("=" * 50)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
overlay = img.copy()
overlay[mask > 0] = (0, 0, 255)                        # paint detected area red
overlay = cv.addWeighted(overlay, 0.5, img, 0.5, 0)
cv.drawContours(overlay, kept, -1, (0, 255, 0), 2)

cv.imwrite(os.path.join(script_dir, "blue_mask.png"), mask)
cv.imwrite(os.path.join(script_dir, "blue_overlay.png"), overlay)
print("Saved blue_mask.png and blue_overlay.png")

# ---------------------------------------------------------------------------
# Display (click the original image to print HSV values for tuning)
# ---------------------------------------------------------------------------
scale = min(1.0, DISPLAY_MAX_WIDTH / img.shape[1])


def resize(im):
    return cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def on_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ox, oy = int(x / scale), int(y / scale)
        h, s, v = hsv[oy, ox]
        b, g, r = img[oy, ox]
        print(f"Clicked ({ox}, {oy}): HSV=({h}, {s}, {v})  RGB=({r}, {g}, {b})")


cv.imshow("Original (click to sample HSV)", resize(img))
cv.imshow("Mask", resize(mask))
cv.imshow("Overlay", resize(overlay))
cv.setMouseCallback("Original (click to sample HSV)", on_click)

cv.waitKey(0)
cv.destroyAllWindows()