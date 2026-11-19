import cv2
import math
import os

# Dossier du script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin vers l'image dans le même dossier que le script
image_path = os.path.join(script_dir, "frame.jpg")

print("Dossier du script :", script_dir)
print("Image utilisée :", image_path)

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Impossible d'ouvrir l'image : {image_path}")

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Point:", x, y)

        if len(points) == 2:
            d = math.sqrt(
                (points[0][0] - points[1][0])**2 +
                (points[0][1] - points[1][1])**2
            )
            print("Distance pixel =", d)

cv2.imshow("image", image)
cv2.setMouseCallback("image", click)

cv2.waitKey(0)
cv2.destroyAllWindows()