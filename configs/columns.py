ID = "Id"
TARGET = "zgon_binary"

MODELING_FEATURES = [
    # --- demografia ---
    "wiek",
    "plec_binary",
    # --- antropometria ---
    "Wzrost",
    "Waga",
    # "bmi",
    # "NumKg",
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
    # "NYHA",
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
    # --- leczenie / procedury ---
    "PCI",
    "CABG",
    "Koronaroplastyka",
    "PCI_lub_zaplanowane_CABG",
    "PCI_i/lub_CABG_pilne",
    "HospKlopidogrelNasycajaca",
    # "kontrast",
    # "dawka",
    "rehabilitacja",
    # --- hospitalizacja ---
    # "dni_pobytu",
    # --- badania laboratoryjne ---
    "alt",
    "aptt",
    "ast",
    "basob",
    "chol",
    # "crp",
    "eosp",
    "fib",
    "gfr",
    # "hba1c",
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
    # "troponin_ratio",
    # "TroponinaT",
    "tsh",
    "tt",
    "ur",
    "nlr",
    "plr",
    # --- inne / techniczne ---
    # "powiazano",
]


PIESZKO2019_ML_FEATURES = [
    # --- demografia / antropometria ---
    "wiek",  # Age
    "plec_binary",  # Sex
    "bmi",  # Body mass index
    # --- pomiary przy przyjęciu ---
    "RRSkurcz",  # Systolic blood pressure
    "RRRozkurcz",  # Diastolic blood pressure
    "EkgHR",  # Heart rate (najbliższy odpowiednik)
    # --- markery hematologiczne / zapalne ---
    "trot",  # Troponin elevation ratio
    "tro",  # Troponin elevation ratio
    # "troponin_ratio", # Troponin elevation ratio
    "nlr",  # Neutrophil to lymphocyte ratio
    "plr",  # Platelet to lymphocyte ratio
    "rdw",  # Red cell distribution width
    "neu",  # Neutrophil count
    "lymb",  # Lymphocyte count (Table 1)
    "plt",  # Platelet count
    "hgb",  # Hemoglobin
    "mcv",  # Mean cell volume
    # --- biochemia / krzepnięcie ---
    "kr",  # Creatinine
    "na",  # Sodium
    "pt",  # Prothrombin time
    "fib",  # Fibrinogen
    "ldl",  # LDL (Table 1)
    "crp",  # C-reactive protein
]

PIESZKO2019_COX_FEATURES = [
    # --- demografia ---
    "wiek",  # Age
    # --- pomiary przy przyjęciu ---
    "EkgHR",  # Heart rate at admission
    "RRSkurcz",  # Systolic blood pressure
    "RRRozkurcz",  # Diastolic blood pressure
    "EkgST",  # Source category for ST-segment elevation
    "dysglikemiaBin",  # Diabetes or prediabetes
    # --- markery hematologiczne / zapalne ---
    "tro",  # TnI component of troponin elevation ratio
    "trot",  # TnT component of troponin elevation ratio
    "crp",  # C-reactive protein
    "nlr",  # Neutrophil to lymphocyte ratio
    "plr",  # Platelet to lymphocyte ratio
    "rdw",  # Red cell distribution width
    "neu",  # Neutrophil count
    "lymb",  # Lymphocyte count
    "monob",  # Monocyte count
    "eosp",  # Eosinophil count
    "basob",  # Basophil count
    "plt",  # Platelet count
    "hgb",  # Hemoglobin
    "hct",  # Hematocrit
    "mcv",  # Mean cell volume
    "mpv",  # Mean platelet volume
    # --- biochemia / krzepnięcie ---
    "kr",  # Creatinine
    "fib",  # Fibrinogen
    "alt",  # Alanine aminotransferase
    "ast",  # Aspartate aminotransferase
    "chol",  # Total cholesterol
    "ldl",  # Low-density lipoprotein
    "hdl",  # High-density lipoprotein
    "trg",  # Triglycerides
    "k",  # Potassium
    "na",  # Sodium
    "ur",  # Urea
    "pt",  # Prothrombin time
]


COLS_OUT = [
    "rehabilitacja",
    "RozpoznanieOZW",
    "SegmentOZW",
    "Rozpoznanie_Glowne",
    "ZatokowyvsInny",
    "DomTypObjawow",
    "CABG",
    "PCI_lub_zaplanowane_CABG",
    "PCI_i/lub_CABG_pilne",
    "Koronaroplastyka",
    "HospKlopidogrelNasycajaca",
    "PCI",
    "WywiadCHNS",
    "WywiadNS",
    "WywiadNerki",
    "WywiadRodzinny",
    "WywiadTObwodowe",
    "LBBB_RBBB",
    "EkgRytm",
    "ast",
    "alt",
    "hdl",
    "chol",
    "ldl",
    "trg",
    "tsh",
    "ur",
    "Cukrzyca",
]


MODELING_FEATURES = [c for c in MODELING_FEATURES if c not in COLS_OUT]


KEEP_AFTER_CLEANING = MODELING_FEATURES + [TARGET]
