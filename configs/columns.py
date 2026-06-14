ID = "Id"
TARGET = "zgon_binary"

MODELING_FEATURES = [
    # --- demografia ---
    "wiek",
    "plec_binary",
    # --- antropometria ---
    "Wzrost",
    "Waga",
    "bmi",
    "NumKg",
    # --- czynniki ryzyka / wywiad ---
    "Cukrzyca",
    "dysglikemiaBin",
    "Nadcisnienie",
    "Palenie",
    "WywiadCHNS",
    "WywiadNS",
    "WywiadNerki",
    "WywiadRodzinny",
    "WywiadTObwodowe",
    "PrzebytyZawal",
    "PrzebyteCABG",
    "PrzebytePTCA",
    # --- stan przyjęcia / objawy ---
    "RRSkurcz",
    "RRRozkurcz",
    "ZatokowyvsInny",
    "NYHA",
    "DomTypObjawow",
    "EkgST",
    "EkgHR",
    "EkgQRS",
    "EkgRytm",
    "LBBB_RBBB",
    # --- rozpoznanie ---
    "RozpoznanieOZW",
    "SegmentOZW",
    "Rozpoznanie_Glowne",
    "TroponinaT",
    # --- leczenie / procedury ---
    "PCI",
    "CABG",
    "Koronaroplastyka",
    "PCI_lub_zaplanowane_CABG",
    "PCI_i/lub_CABG_pilne",
    "HospKlopidogrelNasycajaca",
    "kontrast",
    # "dawka",
    "rehabilitacja",
    # --- hospitalizacja ---
    "dni_pobytu",
    # --- badania laboratoryjne ---
    "alt",
    "aptt",
    "ast",
    "basob",
    "chol",
    "crp",
    "eosp",
    "fib",
    "gfr",
    "hba1c",
    "hct",
    "hdl",
    "hgb",
    "k",
    "kr",
    "ldl",
    "lymb",
    "mcv",
    "monob",
    "mpv",
    "na",
    "neu",
    "plt",
    "pt",
    "rdw",
    "trg",
    "tro",
    "trot",
    "tsh",
    "tt",
    "ur",
    "nlr",
    "plr",
    # --- inne / techniczne ---
    # "powiazano",
]

# Columns persisted after cleaning. By default = features + target; append extra
# columns here to keep them in the saved file without using them as features.
KEEP_AFTER_CLEANING = MODELING_FEATURES + [TARGET]
