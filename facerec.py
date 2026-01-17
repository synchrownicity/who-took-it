
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from supabase import create_client, Client
import datetime
from dotenv import load_dotenv

### weird hacky thing to make everything work
np.int = int

### Supabase Config
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE URL or SUPABASE KEY"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PERSON_TABLE = "Person"
EMBED_TABLE = "Embedding"

### Facial Recognition Model
MODEL_NAME = "buffalo_l"
app = FaceAnalysis(name = MODEL_NAME)  # strong default model pack
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 uses GPU if available, CPU otherwise

### Facial Similarity Function
def face_similarity(embed_one: np.ndarray, embed_two: np.ndarray, threshold: float) -> tuple[float, bool]:
    similarity = np.dot(embed_one, embed_two) / (np.linalg.norm(embed_one) * np.linalg.norm(embed_two))
    return similarity, similarity > threshold

### Create a person
def create_person() -> str:
    res = supabase.table(PERSON_TABLE).insert({}).execute()

    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase insert(Person) error: {res.error}")

    return res.data[0]["id"]

### Save the Embedding
def save_embedding(person_id: str, embedding: np.ndarray) -> None:
    payload = {
        "person_id": person_id, 
        "vector": embedding.astype(np.float32).tolist(),
    }

    res = supabase.table(EMBED_TABLE).insert(payload).execute()

    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase insert error: {res.error}")

### Load the Embeddings
def load_embeddings() -> list[tuple[str, np.ndarray]]:
    res = supabase.table(EMBED_TABLE).select("person_id, vector").execute()
    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase select(Embedding) error: {res.error}")
    
    out: list[tuple[str, np.ndarray]] = []
    
    for row in (res.data or []):
        pid = row["person_id"]
        vect = row["vector"]

        if vect is None:
            continue

        out.append((pid, np.array(vect, dtype = np.float32)))

    return out

### Match the Faces
def best_match(cur_embedding: np.ndarray, db: list[tuple[str, np.ndarray]], threshold: float) -> tuple[str | None, float]:
    best_person_id = None
    best_sim = -1.0

    for person_id, db_embedding in db:
        similarity, _ = face_similarity(cur_embedding, db_embedding, threshold)

        if similarity > best_sim:
            best_sim = similarity
            best_person_id = person_id
        
    if best_sim >= threshold:
        return best_person_id, best_sim
        
    return None, best_sim

### Image Captured by Frontend
capt_img_file_path = "emilia_3.jpg"
capt_img = cv2.imread(capt_img_file_path)
capt_faces = app.get(capt_img)
print(f"Found {len(capt_faces)} face(s).")

# Load DB embeddings + compare
db = load_embeddings()

THRESHOLD = 0.50

for i, face in enumerate(capt_faces):
    probe = face.embedding
    person_id, similarity = best_match(probe, db, THRESHOLD)  

    if person_id is None:
        print(f"[Face #{i}] Unknown (best similarity = {similarity: .3f})")

        # Create a new person ONCE
        new_person_id = create_person()
        save_embedding(new_person_id, probe)

        print(f"[Face {i}] Enrolled new person = {new_person_id}")

    else: 
        print(f"[Face {i}] Matched person_id = {person_id} (similarity = {similarity: .3f})")

        save_embedding(person_id, probe)


### Bounding Box
#out = app.draw_on(img, faces)
#out_path = "89_output.jpg"
#cv2.imwrite(out_path, out)
#print("Image saved.")