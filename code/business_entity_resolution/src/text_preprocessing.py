"""
Text Preprocessing & Normalization Module
Part of Phase 2 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

CRITICAL SPECIFICATIONS:
- Country-agnostic cleaning for business_name and business_address.
- Handles US, Indian, and French legal suffixes/prefixes (Inc, Corp, Pvt Ltd, SARL, SAS, SCI, EURL, etc.).
- Normalizes punctuation, lowercasing, unicode accents (e.g., é, ç, à, ô), and address abbreviations (Rd -> Road, St -> Street, Ste -> Suite, etc.).
- Extracts numeric components (ZIP/PIN codes, building numbers) into separate auxiliary features.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

# ---------------------------------------------------------------------------
# Unicode Normalization & Accent Handling
# ---------------------------------------------------------------------------

LIGATURE_MAP = {
    "œ": "oe",
    "Œ": "OE",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
}


def strip_accents(text: str) -> str:
    """
    Strips Latin diacritical marks (e.g., é -> e, ç -> c, ô -> o, à -> a)
    while safely preserving non-Latin scripts (e.g. Devanagari / Indic characters).
    Also unpacks standard European ligatures.
    """
    if not text:
        return ""

    for lig, repl in LIGATURE_MAP.items():
        if lig in text:
            text = text.replace(lig, repl)

    # Decompose into base characters + combining diacritics (NFKD)
    decomposed = unicodedata.normalize("NFKD", text)

    # Filter out Latin combining diacritical marks (Unicode range 0x0300 to 0x036F)
    cleaned = "".join(
        ch for ch in decomposed if not (0x0300 <= ord(ch) <= 0x036F)
    )
    return unicodedata.normalize("NFC", cleaned)


# ---------------------------------------------------------------------------
# Legal Suffixes & Prefixes (Country-Agnostic)
# ---------------------------------------------------------------------------

# Ordered from multi-word to single-word to ensure greedy matching
LEGAL_ENTITIES = [
    # Multi-word forms
    (r"\bprivate\s+limited\b", "pvt_ltd"),
    (r"\bpvt\s*\.?\s*ltd\s*\.?", "pvt_ltd"),
    (r"\bpvt\s*\.?\s*limited\b", "pvt_ltd"),
    (r"\bprivate\s+ltd\s*\.?", "pvt_ltd"),
    (r"\bpublic\s+limited\b", "pub_ltd"),
    (r"\bpub\s*\.?\s*ltd\s*\.?", "pub_ltd"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bgeneral\s+partnership\b", "gp"),
    (r"\bsole\s+proprietorship\b", "prop"),
    (r"\bproprietorship\b", "prop"),
    (r"\bjoint\s+venture\b", "jv"),
    (r"\band\s+sons\b", "sons"),
    (r"\b&\s*sons\b", "sons"),
    (r"\band\s+associates\b", "associates"),
    (r"\b&\s*associates\b", "associates"),
    (r"\bet\s+fils\b", "fils"),
    (r"\b&\s*fils\b", "fils"),
    
    # French / European forms
    (r"\bsociete\s+a\s+responsabilite\s+limitee\b", "sarl"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\b", "sas"),
    (r"\bsociete\s+anonyme\b", "sa"),
    (r"\bsociete\s+civile\s+immobiliere\b", "sci"),
    (r"\bentreprise\s+unipersonnelle\b", "eurl"),
    (r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\s*\.?", "sarl"),
    (r"\bsarlu\b", "sarl"),
    (r"\bs\s*\.?\s*a\s*\.?\s*s\s*\.?\s*u\s*\.?", "sasu"),
    (r"\bs\s*\.?\s*a\s*\.?\s*s\s*\.?", "sas"),
    (r"\bs\s*\.?\s*a\s*\.?", "sa"),
    (r"\be\s*\.?\s*u\s*\.?\s*r\s*\.?\s*l\s*\.?", "eurl"),
    (r"\bs\s*\.?\s*c\s*\.?\s*i\s*\.?", "sci"),
    (r"\bs\s*\.?\s*n\s*\.?\s*c\s*\.?", "snc"),
    (r"\bg\s*\.?\s*i\s*\.?\s*e\s*\.?", "gie"),
    (r"\be\s*\.?\s*i\s*\.?\s*r\s*\.?\s*l\s*\.?", "eirl"),
    (r"\be\s*\.?\s*i\s*\.?", "ei"),
    (r"\bassociation\b", "asso"),
    (r"\bassn\b", "asso"),
    (r"\bcie\b", "co"),
    (r"\bcompagnie\b", "co"),
    
    # Standard single-word forms
    (r"\binc\s*\.?", "inc"),
    (r"\bcorp\s*\.?", "corp"),
    (r"\bllc\b", "llc"),
    (r"\bl\s*\.?\s*l\s*\.?\s*c\s*\.?", "llc"),
    (r"\bllp\b", "llp"),
    (r"\bl\s*\.?\s*l\s*\.?\s*p\s*\.?", "llp"),
    (r"\bltd\s*\.?", "ltd"),
    (r"\blimited\b", "ltd"),
    (r"\bco\s*\.?", "co"),
    (r"\bcompany\b", "co"),
    (r"\bplc\b", "plc"),
    (r"\bpllc\b", "pllc"),
    (r"\bgmbh\b", "gmbh"),
    (r"\bopc\b", "opc"),
]

# Compile patterns for fast vector and string operations
COMPILED_LEGAL_ENTITIES = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in LEGAL_ENTITIES
]


def extract_legal_form(name: str) -> Tuple[str, str]:
    """
    Extracts the legal entity type (whether prefix, suffix, or bracketed)
    and returns a tuple of (clean_name_without_legal_form, canonical_legal_form).
    """
    if not name:
        return "", ""

    text = name.strip()
    extracted_form = ""

    # Check for bracketed legal form e.g. "QHC Culture [EURL]"
    bracket_match = re.search(r"\[([a-zA-Z\s\.]+)\]|\(([a-zA-Z\s\.]+)\)", text)
    if bracket_match:
        inner = (bracket_match.group(1) or bracket_match.group(2) or "").strip().lower()
        for pattern, canonical in COMPILED_LEGAL_ENTITIES:
            if pattern.search(inner):
                extracted_form = canonical
                text = text[:bracket_match.start()] + " " + text[bracket_match.end():]
                break

    # Check for prefix legal form e.g. "SCI Ptit Amicale", "SARL Dupont"
    for pattern, canonical in COMPILED_LEGAL_ENTITIES:
        match = pattern.match(text)
        if match:
            if not extracted_form:
                extracted_form = canonical
            text = text[match.end():].strip()
            break

    # Check for suffix legal form e.g. "Zephay Labs Inc", "Om Constructions Pvt Ltd"
    for pattern, canonical in COMPILED_LEGAL_ENTITIES:
        # Search at the end of string
        match = list(pattern.finditer(text))
        if match:
            last_match = match[-1]
            # Ensure it is at or near the end (allowing punctuation/whitespace)
            remaining = text[last_match.end():].strip(" .,-")
            if not remaining:
                if not extracted_form:
                    extracted_form = canonical
                text = text[:last_match.start()].strip()
                break

    # Clean any dangling punctuation or whitespace
    clean_name = re.sub(r"^[^\w]+|[^\w]+$", "", text).strip()
    return clean_name, extracted_form


# ---------------------------------------------------------------------------
# Address Normalization & Expansion
# ---------------------------------------------------------------------------

# Global dictionary of address component abbreviations mapped to standard canonical forms
ADDRESS_ABBREVIATIONS = {
    # Street Types (US / Global)
    r"\bst\b|\bst\.": "street",
    r"\brd\b|\brd\.": "road",
    r"\bave\b|\bav\b|\bave\.|\bav\.": "avenue",
    r"\bblvd\b|\bbvd\b|\bbld\b|\bblvd\.": "boulevard",
    r"\bdr\b|\bdr\.": "drive",
    r"\bln\b|\bln\.": "lane",
    r"\bct\b|\bct\.": "court",
    r"\bpl\b|\bpl\.": "place",
    r"\bsq\b|\bsq\.": "square",
    r"\bcir\b|\bcir\.": "circle",
    r"\bhwy\b|\bhwy\.": "highway",
    r"\bpkwy\b|\bpkwy\.": "parkway",
    r"\bter\b|\bterr\b": "terrace",
    r"\btr\b|\btrl\b": "trail",
    r"\bway\b": "way",
    r"\bpk\b": "park",
    
    # Secondary Unit Types
    r"\bste\b|\bste\.": "suite",
    r"\bapt\b|\bapt\.|\bapart\b|\bapartment\b": "apartment",
    r"\bunit\b": "unit",
    r"\bbldg\b|\bbldg\.": "building",
    r"\bfl\b|\bfl\.|\bflr\b": "floor",
    r"\brm\b|\brm\.": "room",
    r"\bno\b|\bno\.": "number",
    
    # French Address Elements
    r"\br\b|\br\.": "rue",
    r"\bbd\b|\bbd\.": "boulevard",
    r"\ball\b|\ball\.": "allee",
    r"\bimp\b|\bimp\.": "impasse",
    r"\bch\b|\bch\.|\bche\b": "chemin",
    r"\brte\b|\brt\b": "route",
    r"\bcrs\b": "cours",
    r"\bqu\b": "quai",
    r"\bbis\b|\bb\b": "bis",
    r"\bter\b|\bt\b": "ter",
    r"\bres\b|\bresid\b": "residence",
    r"\bbat\b|\bbat\.": "batiment",
    
    # Indian Address Elements
    r"\bopp\b|\bopp\.": "opposite",
    r"\bnr\b|\bnr\.": "near",
    r"\badj\b|\badj\.": "adjacent",
    r"\bsec\b|\bsec\.": "sector",
    r"\bcol\b|\bcol\.": "colony",
    r"\bext\b|\bextn\b": "extension",
    r"\bblk\b|\bblk\.": "block",
    r"\bflt\b|\bflt\.": "flat",
    r"\bnag\b|\bngr\b": "nagar",
    r"\bmkt\b": "market",
    r"\bbzr\b|\bbzn\b": "bazaar",
    r"\bh\.?\s*no\.?": "house number",
    r"\bplt\.?\s*no\.?|\bplot\s+no\.?": "plot number",
}

COMPILED_ADDRESS_ABBREVIATIONS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in ADDRESS_ABBREVIATIONS.items()
]


def normalize_numbers(text: str) -> str:
    """
    Normalizes leading zeros in alphanumeric tokens (e.g. K-00303 -> K-303, Plot 007 -> Plot 7).
    """
    # Leading zeros in alphanumeric hyphenated tokens like K-00303
    text = re.sub(r"(?<=[a-zA-Z\-])0+(\d+)", r"\1", text)
    # Standalone leading zeros like 007 -> 7
    text = re.sub(r"\b0+(\d+)\b", r"\1", text)
    return text


def clean_text_basic(text: str) -> str:
    """
    Applies lowercasing, unicode unaccenting, and basic punctuation normalization.
    """
    if not text:
        return ""
    text = str(text)
    # Strip Latin accents & normalize ligatures
    text = strip_accents(text).lower()
    # Replace '&' with 'and'
    text = re.sub(r"&", " and ", text)
    # Normalize noise zero padding
    text = normalize_numbers(text)
    # Replace punctuation with spaces except characters in Indic scripts
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name: str) -> Dict[str, str]:
    """
    Cleans a business name in a country-agnostic manner.

    Returns:
    - 'name_clean': Cleaned core name with legal forms removed and punctuation stripped.
    - 'name_raw_clean': Cleaned name preserving legal forms.
    - 'legal_form': Extracted canonical legal entity form (e.g., 'inc', 'pvt_ltd', 'sarl', 'llp').
    """
    if not name:
        return {"name_clean": "", "name_raw_clean": "", "legal_form": ""}

    text = str(name).strip()
    # Strip accents first to ensure accent-injected legal forms match
    text_unaccented = strip_accents(text)

    # Extract legal form
    core_name, legal_form = extract_legal_form(text_unaccented)

    name_clean = clean_text_basic(core_name)
    name_raw_clean = clean_text_basic(text)

    return {
        "name_clean": name_clean,
        "name_raw_clean": name_raw_clean,
        "legal_form": legal_form,
    }


def clean_business_address(address: str) -> Dict[str, Union[str, List[str]]]:
    """
    Cleans and expands a business address in a country-agnostic manner.

    Returns:
    - 'address_clean': Normalized address string with expanded standard abbreviations.
    - 'address_tokens': Ordered list of non-trivial words/tokens.
    """
    if not address:
        return {"address_clean": "", "address_tokens": []}

    text = str(address).strip()
    # Strip Latin accents and normalize zeros
    text = strip_accents(text).lower()
    text = re.sub(r"&", " and ", text)
    text = normalize_numbers(text)

    # Expand address abbreviations
    for pattern, repl in COMPILED_ADDRESS_ABBREVIATIONS:
        text = pattern.sub(f" {repl} ", text)

    # Normalize punctuation and whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in text.split() if len(t) > 0]

    return {
        "address_clean": text,
        "address_tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Numeric Features Extraction (Postal Codes & Building Numbers)
# ---------------------------------------------------------------------------

PIN_CODE_PATTERN = re.compile(r"\b[1-9][0-9]{5}\b")            # India: 6 digits
ZIP_CODE_PATTERN = re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b")   # US: 5 digits or 5+4
FR_POSTAL_PATTERN = re.compile(r"\b(?:0[1-9]|[1-8][0-9]|9[0-8])[0-9]{3}\b")  # France: 5 digits (01-98)
BUILDING_NUM_PATTERN = re.compile(r"\b(?:\d+[a-zA-Z]?|[a-zA-Z]-?\d+)\b")


def extract_numeric_features(
    address: str,
    country: Optional[str] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Extracts numeric components from an address string:
    - postal_code: Detected PIN/ZIP/Postal code (handles India 6-digit, US 5-digit, France 5-digit).
    - building_numbers: Extracted house/building numbers with leading zeros stripped.
    - all_numbers: All isolated numeric tokens.
    """
    if not address:
        return {"postal_code": "", "building_numbers": [], "all_numbers": []}

    text = normalize_numbers(str(address))
    c_upper = (country or "").strip().upper()

    postal_code = ""

    # Country-informed or heuristic postal code detection
    if c_upper == "INDIA":
        pin_match = PIN_CODE_PATTERN.findall(text)
        if pin_match:
            postal_code = pin_match[-1]  # Postal codes usually appear at the end
    elif c_upper == "US":
        zip_match = ZIP_CODE_PATTERN.findall(text)
        if zip_match:
            postal_code = zip_match[-1]
    elif c_upper == "FRANCE":
        fr_match = FR_POSTAL_PATTERN.findall(text)
        if fr_match:
            postal_code = fr_match[-1]
    else:
        # Heuristic search across all patterns
        all_pins = PIN_CODE_PATTERN.findall(text)
        all_zips = ZIP_CODE_PATTERN.findall(text)
        if all_pins:
            postal_code = all_pins[-1]
        elif all_zips:
            postal_code = all_zips[-1]

    # Extract building / street numbers
    # Remove the detected postal code first so it isn't confused with building number
    text_no_postal = text
    if postal_code:
        text_no_postal = text_no_postal.replace(postal_code, " ")

    # Find numbers and alphanumeric building designators (e.g. 220A, K-303, 1064)
    all_num_tokens = re.findall(r"\b\d+\b", text_no_postal)
    building_candidates = BUILDING_NUM_PATTERN.findall(text_no_postal)

    # Filter building numbers: usually <= 5 digits, not purely letters
    building_numbers = []
    for cand in building_candidates:
        if cand == postal_code:
            continue
        # Exclude common noise words like 1st, 2nd, 3rd, 4th
        if cand.lower() in {"1st", "2nd", "3rd", "4th", "5th"}:
            continue
        if any(char.isdigit() for char in cand):
            building_numbers.append(cand.lower())

    return {
        "postal_code": postal_code,
        "building_numbers": building_numbers,
        "all_numbers": all_num_tokens,
    }


# ---------------------------------------------------------------------------
# Vectorized / Batch Processing for DataFrames
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Applies full text normalization and feature extraction to a DataFrame.
    Adds:
    - clean_name: core name (lowercase, unaccented, legal form stripped)
    - name_raw_clean: name (lowercase, unaccented, legal form preserved)
    - legal_form: canonical legal form tag
    - clean_address: normalized address with expanded abbreviations
    - postal_code: extracted postal/PIN code
    - building_numbers: list of extracted building/house numbers
    """
    res = df if inplace else df.copy()

    # Preprocess business names
    name_results = [clean_business_name(n) for n in res["business_name"]]
    res["clean_name"] = [r["name_clean"] for r in name_results]
    res["name_raw_clean"] = [r["name_raw_clean"] for r in name_results]
    res["legal_form"] = [r["legal_form"] for r in name_results]

    # Preprocess addresses
    addr_results = [clean_business_address(a) for a in res["business_address"]]
    res["clean_address"] = [r["address_clean"] for r in addr_results]

    # Extract numeric components
    countries = res["country"] if "country" in res.columns else [""] * len(res)
    num_results = [
        extract_numeric_features(a, c)
        for a, c in zip(res["business_address"], countries)
    ]
    res["postal_code"] = [r["postal_code"] for r in num_results]
    res["building_numbers"] = [r["building_numbers"] for r in num_results]

    return res


# Aliases for module interface compatibility
normalize_text = clean_text_basic
extract_legal_suffix = extract_legal_form

