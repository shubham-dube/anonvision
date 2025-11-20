from typing import List, Dict, Optional
import re


def _lowerwords(s: str) -> List[str]:
    """Helper: split and lowercase a prompt string into words/tokens."""
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


def _has_role(person: Dict, role_name: str) -> bool:
    """Check if a person has a specific role (e.g., teacher, student)."""
    attrs = person.get("attributes") or {}
    role = attrs.get("role") or attrs.get("job") or attrs.get("position")

    if isinstance(role, str) and role.lower() == role_name:
        return True

    # handle flags like is_teacher=True
    if attrs.get(f"is_{role_name}") or attrs.get(role_name):
        return True

    return False


def prompt_to_filter(prompt: str):
    """
    Convert a user prompt into a filter function f(person) -> bool (True = blur).
    Returns None if prompt unrecognized.
    """
    if not prompt or not prompt.strip():
        return None

    words = _lowerwords(prompt)
    ws = set(words)

    # Universal filters
    if {"everyone", "all"} & ws:
        return lambda p: True
    if {"nobody", "none"} & ws or "no one" in prompt.lower():
        return lambda p: False

    # Role filters
    if {"student", "students"} & ws:
        return lambda p: not _has_role(p, "teacher")
    if {"teacher", "teachers"} & ws:
        return lambda p: _has_role(p, "teacher")

    # Face filters
    if {"face", "faces"} & ws:
        if {"no", "not"} & ws:
            return lambda p: not bool(p.get("bbox_face"))
        return lambda p: bool(p.get("bbox_face"))

    # Color filters
    color_keywords = {
        "red", "blue", "green", "black", "white", "yellow",
        "brown", "pink", "orange", "purple", "gray", "grey"
    }
    matched_colors = [c for c in color_keywords if c in ws]
    if matched_colors:
        color = matched_colors[0]
        return lambda p: (
            isinstance(p.get("dress_color"), str)
            and color in p["dress_color"].lower()
        )

    # Pose filters
    pose_options = {"sitting", "standing", "running", "walking", "lying", "unknown"}
    matched_poses = [pose for pose in pose_options if pose in ws]
    if matched_poses:
        pose = matched_poses[0]
        return lambda p: (
            isinstance(p.get("pose"), str)
            and p["pose"].lower() == pose
        )

    # Clothing filters
    clothing_tokens = {"mask", "masked", "hood", "hat", "cap", "helmet"}
    if ws & clothing_tokens:
        return lambda p: (
            isinstance(p.get("cloth"), str)
            and any(tok in p["cloth"].lower() for tok in clothing_tokens)
        )

    # Person ID filters (e.g., "blur person 2 and 4")
    matches = re.findall(r"\bperson\s*(\d+)\b", prompt.lower())
    if matches:
        ids = {int(x) for x in matches}
        return lambda p: p.get("id") in ids

    # Too generic or unrecognized prompt
    if {"blur", "hide", "anonymize", "censor"} & ws:
        return None

    return None


def environment_default_filter(environment: Optional[str]):
    """Return safe default filter function based on environment."""
    if not environment:
        return lambda p: bool(p.get("bbox_face"))

    env = environment.strip().lower()

    if env in {"classroom", "lecture", "school"}:
        # Blur students
        return lambda p: not _has_role(p, "teacher")
    if env in {"meeting", "conference", "boardroom"}:
        # Blur audience, not speakers
        return lambda p: not (
            _has_role(p, "presenter")
            or _has_role(p, "host")
            or _has_role(p, "teacher")
        )
    if env in {"public", "street", "market"}:
        return lambda p: bool(p.get("bbox_face"))
    if env in {"private", "home"}:
        return lambda p: False

    return lambda p: bool(p.get("bbox_face"))


def select_ids_to_blur(
    persons: List[Dict],
    prompt: Optional[str] = None,
    environment: Optional[str] = None
) -> List[int]:
    """Main logic to return IDs of persons to blur."""
    if not isinstance(persons, list) or not all(isinstance(p, dict) for p in persons):
        raise ValueError("Input must be a list of person dictionaries.")

    if prompt:
        filter_fn = prompt_to_filter(prompt)
        if filter_fn is None:
            return []
        return [p.get("id") for p in persons if safe_eval(filter_fn, p)]

    # Fallback to environment
    filter_fn = environment_default_filter(environment)
    return [p.get("id") for p in persons if safe_eval(filter_fn, p)]


def safe_eval(func, person):
    """Safely evaluate filter function to avoid runtime errors."""
    try:
        return bool(func(person))
    except Exception:
        return False


# ---------- Quick Test ----------
if __name__ == "__main__":
    sample = {
        "context": "teacher teaching students in a classroom",
        "persons": [
            {"id": 1, "bbox_face": [1, 1, 5, 5],
             "attributes": {"role": "teacher"}, "cloth": "shirt", "dress_color": "blue", "pose": "standing"},
            {"id": 2, "bbox_face": [11, 1, 15, 5],
             "attributes": {"role": "student"}, "cloth": "hoodie", "dress_color": "red", "pose": "sitting"},
            {"id": 3, "bbox_face": None,
             "attributes": {"role": "student"}, "cloth": "jacket", "dress_color": "green", "pose": "sitting"},
            {"id": 4, "bbox_face": [31, 1, 35, 5],
             "attributes": {"role": "staff"}, "cloth": "t-shirt", "dress_color": "black", "pose": "walking"}
        ]
    }

    print("Prompt: 'blur students' ->", select_ids_to_blur(sample["persons"], prompt="blur students"))
    print("Environment: 'classroom' ->", select_ids_to_blur(sample["persons"], environment="classroom"))
