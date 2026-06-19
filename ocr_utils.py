from pathlib import Path
from typing import List
from collections import Counter
import difflib
import re
import cv2

try:
    from fast_plate_ocr import LicensePlateRecognizer
    OCR_AVAILABLE = True
except Exception:
    LicensePlateRecognizer = None
    OCR_AVAILABLE = False

_ocr = None

PRIMARY_SERIES = {"AA", "AB", "AC", "AD", "AE"}
ALLOWED_WILAYAS = {f"{i:02d}" for i in range(1, 13)}

DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"})
LETTER_FIX = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"})

RE_NORMAL = re.compile(r"^\d{4}[A-Z]{2}\d{2}$")
RE_STATE = re.compile(r"^(SG|SP|SCC)\d{5}$")
RE_IF = re.compile(r"^\d{5}IF$")
RE_TT = re.compile(r"^[A-Z0-9]\d{5}TT$")
RE_IT = re.compile(r"^IT\d{4}$")
RE_ONU = re.compile(r"^ONU\d{4}(CMD)?$")
RE_DIPLO = re.compile(r"^\d{2,3}(CD|CC|CMD)\d{4}$")
RE_WT = re.compile(r"^WT\d{5}$")
RE_ZFN = re.compile(r"^ZFN\d{5}$")
LEGAL_REGEXES = [RE_NORMAL, RE_STATE, RE_IF, RE_TT, RE_IT, RE_ONU, RE_DIPLO, RE_WT, RE_ZFN]


def clean_text(t: str) -> str:
    t = t.upper().strip().replace("_", "").replace(" ", "").replace("-", "")
    return re.sub(r"[^A-Z0-9]", "", t)


def legal_type(s: str) -> str:
    if RE_NORMAL.fullmatch(s): return "NORMAL"
    if RE_STATE.fullmatch(s): return "STATE"
    if RE_IF.fullmatch(s): return "IF"
    if RE_TT.fullmatch(s): return "TT"
    if RE_IT.fullmatch(s): return "IT"
    if RE_ONU.fullmatch(s): return "ONU"
    if RE_DIPLO.fullmatch(s): return "DIPLO"
    if RE_WT.fullmatch(s): return "WT"
    if RE_ZFN.fullmatch(s): return "ZFN"
    return "UNK"


def is_legal_plate(s: str) -> bool:
    if not s:
        return False
    for r in LEGAL_REGEXES:
        if r.fullmatch(s):
            if r is RE_NORMAL:
                series = s[4:6]
                suffix = s[6:8]
                if series in PRIMARY_SERIES:
                    return suffix == "00" or suffix in ALLOWED_WILAYAS
                return True
            return True
    return False


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_legal_substrings(s: str) -> List[str]:
    cands = set()
    for length in range(6, 13):
        for i in range(0, max(0, len(s) - length + 1)):
            sub = s[i:i + length]
            if is_legal_plate(sub):
                cands.add(sub)
    return list(cands)


def base_readability_score(s: str) -> int:
    if not s:
        return -10000
    score = 0
    score -= abs(len(s) - 8) * 3
    if len(s) < 6:
        score -= 80
    if len(s) > 12:
        score -= 40
    repeated = sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
    score -= repeated * 2
    score -= (s.count("O") + s.count("Q") + s.count("I") + s.count("L")) * 2
    return score


def law_bonus(s: str, support: int = 1) -> int:
    if not is_legal_plate(s):
        return 0
    t = legal_type(s)
    if t in {"STATE", "TT", "DIPLO", "ONU", "IT", "IF", "WT", "ZFN"}:
        return 120 + min(40, (support - 1) * 20)
    if t == "NORMAL":
        series = s[4:6]
        suffix = s[6:8]
        if series in PRIMARY_SERIES and (suffix == "00" or suffix in ALLOWED_WILAYAS):
            return 160 + min(40, (support - 1) * 20)
        if suffix == "00":
            return 40 + min(40, (support - 1) * 20)
    return 0


def final_score(s: str, support: int = 1) -> int:
    return support * 25 + base_readability_score(s) + law_bonus(s, support=support)


def normalize_by_template(s: str) -> str:
    if not s:
        return s
    if len(s) == 8:
        return s[:4].translate(DIGIT_FIX) + s[4:6].translate(LETTER_FIX) + s[6:8].translate(DIGIT_FIX)
    if s.startswith(("SG", "SP")) and len(s) == 7:
        return s[:2] + s[2:].translate(DIGIT_FIX)
    if s.startswith("SCC") and len(s) == 8:
        return s[:3] + s[3:].translate(DIGIT_FIX)
    if s.endswith("IF") and len(s) == 7:
        return s[:5].translate(DIGIT_FIX) + "IF"
    if s.endswith("TT") and len(s) == 7:
        return s[0].translate(LETTER_FIX) + s[1:6].translate(DIGIT_FIX) + "TT"
    if s.startswith("WT") and len(s) == 7:
        return "WT" + s[2:].translate(DIGIT_FIX)
    if s.startswith("ZFN") and len(s) == 8:
        return "ZFN" + s[3:].translate(DIGIT_FIX)
    if s.startswith("IT") and len(s) == 6:
        return "IT" + s[2:].translate(DIGIT_FIX)
    if s.startswith("ONU") and len(s) == 7:
        return "ONU" + s[3:].translate(DIGIT_FIX)
    for token in ("CD", "CC", "CMD"):
        j = s.find(token)
        if j in (2, 3) and len(s) >= j + len(token) + 4:
            return s[:j].translate(DIGIT_FIX) + token + s[j + len(token):j + len(token) + 4].translate(DIGIT_FIX)
    return s


def generate_repair_candidates(cleaned_outputs: List[str]) -> List[str]:
    cands, seen = [], set()
    for s in cleaned_outputs:
        for x in [s] + extract_legal_substrings(s):
            x = normalize_by_template(x)
            if x and x not in seen:
                seen.add(x)
                cands.append(x)
    return cands[:250]


def pick_best_with_law(cleaned_outputs: List[str]) -> str:
    if not cleaned_outputs:
        return ""
    counts = Counter(cleaned_outputs)
    legal_direct = [s for s in counts if is_legal_plate(s)]
    if legal_direct:
        primary, special, other_supported = [], [], []
        for s in legal_direct:
            t = legal_type(s)
            if t == "NORMAL":
                series = s[4:6]
                suffix = s[6:8]
                if series in PRIMARY_SERIES and (suffix == "00" or suffix in ALLOWED_WILAYAS):
                    primary.append(s)
                elif counts[s] >= 2:
                    other_supported.append(s)
            else:
                special.append(s)
        if primary:
            return max(primary, key=lambda x: final_score(x, counts[x]))
        if special:
            return max(special, key=lambda x: final_score(x, counts[x]))
        if other_supported:
            return max(other_supported, key=lambda x: final_score(x, counts[x]))

    clusters = []
    for s in cleaned_outputs:
        placed = False
        for cluster in clusters:
            if similarity(s, cluster[0]) >= 0.82:
                cluster.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])

    reps = [max(cluster, key=lambda x: (len(x), base_readability_score(x))) for cluster in clusters]
    candidates = generate_repair_candidates(list(cleaned_outputs) + reps)
    return max(candidates, key=lambda x: final_score(x, counts.get(x, 1))) if candidates else ""


def recognize_plate(crop_bgr, tmp_dir: Path) -> str:
    global _ocr
    if not OCR_AVAILABLE:
        return "NO_OCR_LIB"
    if _ocr is None:
        _ocr = LicensePlateRecognizer("global-plates-mobile-vit-v2-model")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / "__tmp_plate.png"
    cv2.imwrite(str(tmp_path), crop_bgr)

    raw = []
    try:
        result = _ocr.run(str(tmp_path))
        if result:
            raw.extend(str(x) for x in result[:3])
    except Exception:
        raw = []

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    cleaned = [clean_text(x) for x in raw]
    cleaned = [x for x in cleaned if x]
    return pick_best_with_law(cleaned) if cleaned else "NO_OCR"
