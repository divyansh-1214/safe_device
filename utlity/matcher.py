# utlity/matcher.py
# Module 5 — Face Matching
# Compares a live embedding against stored embeddings using cosine similarity.
# Both embeddings must be L2-normalized (norm == 1.0) for cosine sim = dot product.

import os
import pickle
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "face_db", "faces.pkl")


class FaceMatcher:
    """Load a face database from disk and match live embeddings against it.

    Database format (pickle file):
        {
            "Divyansh": np.ndarray of shape (512,),
            ...
        }
    """

    THRESHOLD = 0.4   # cosine similarity threshold (both vectors must be unit-norm)

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._db: dict[str, np.ndarray] = {}
        self.reload()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """(Re)load the face database from disk. Safe to call if file is missing."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self._db = pickle.load(f)
            print(f"[FaceMatcher] Loaded {len(self._db)} face(s) from {self.db_path}")
        else:
            self._db = {}
            print("[FaceMatcher] No database found — starting empty.")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self._db, f)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Store (or overwrite) a named embedding and persist to disk."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm                  # ensure unit norm before saving
        self._db[name] = emb
        self._save()
        print(f"[FaceMatcher] Saved '{name}' → {self.db_path}")

    def remove(self, name: str) -> bool:
        """Delete a person from the database. Returns True if found."""
        if name in self._db:
            del self._db[name]
            self._save()
            print(f"[FaceMatcher] Removed '{name}'")
            return True
        print(f"[FaceMatcher] '{name}' not found in database")
        return False

    def list_names(self) -> list[str]:
        return list(self._db.keys())

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Compare embedding against database.

        Args:
            embedding: L2-normalized (512,) float32 array.

        Returns:
            (name, score) — name is "Unknown" if best score < THRESHOLD.
        """
        if not self._db:
            return "Unknown", 0.0

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_name = "Unknown"
        best_score = -1.0

        for name, known_emb in self._db.items():
            # Both vectors are unit-norm → cosine sim == dot product
            score = float(np.dot(emb, known_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score
