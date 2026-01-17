
import cv2
import numpy as np
from insightface.app import FaceAnalysis

np.int = int
app = FaceAnalysis(name="buffalo_l")  # strong default model pack
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 uses GPU if available, CPU otherwise

img_path = "70.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(
        f"Could not read '{img_path}'. "
        f"Check the file path and your working directory."
    )

faces = app.get(img)
#rimg = app.draw_on(img, faces)
#cv2.imwrite("./t1_output.jpg", rimg)

print(f"Found {len(faces)} face(s) in {img_path}")

embedded_face = faces[0].embedding

def face_similarity(embed_one: np.ndarray, embed_two: np.ndarray, threshold: float) -> tuple[float, bool]:
    similarity = np.dot(embed_one, embed_two) / (np.linalg.norm(embed_one) * np.linalg.norm(embed_two))

    return similarity, similarity > threshold

image_paths = ["two_faces.jpg"]

for path in image_paths:
    cur_image = cv2.imread(path)

    if cur_image is None:
        raise FileNotFoundError(
        f"Could not read '{img_path}'. "
        f"Check the file path and your working directory."
    )

    faces = app.get(cur_image)

    print(f"There are {len(faces)} faces in the picture.")

    if len(faces) <= 0:   # if there are no faces
        continue

    if len(faces) >= 1:
        embedded_face_arr = []
        for i in range(len(faces)):
            embedded_face_arr.append(faces[i].embedding)

    for embedded_face_i in embedded_face_arr:
        similarity, is_same_face = face_similarity(embedded_face_i, embedded_face, 0.50)
        print(similarity, is_same_face)

### Bounding Box
#out = app.draw_on(img, faces)
#out_path = "89_output.jpg"
#cv2.imwrite(out_path, out)
#print("Image saved.")