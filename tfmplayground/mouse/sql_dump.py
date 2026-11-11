"""Read a MySQL dump of the mouse database into a dict of DataFrames.

The result has the shape of ``/data/mice/maus.pandas.pkl``: one DataFrame per
table, columns in ``CREATE TABLE`` order, a RangeIndex in dump order. Values
keep the representation of that pickle: dates and times are strings (zero
dates such as ``0000-00-00`` stay as they are), numbers are int or float as
written in the dump, ``bit`` values are ints and ``NULL`` is missing.

With ``like`` (e.g. the old pickle), shared columns are cast to its dtypes
where this loses nothing, ``Visiten.Zeitpunkt`` is derived as in the old
pickle, and the date/datetime/timestamp columns that the old pickle does not
have become datetime64. The existing loaders then work unchanged::

    python -m tfmplayground.mouse.sql_dump

checks the parser on the old dump (it must reproduce the old pickle) and then
converts ``mausprojekt-3.sql`` to ``mausprojekt-3.pandas.pkl``. With
``--english`` it instead writes ``mausprojekt-3.en.pandas.pkl``, an English copy
of that pickle (:func:`translate`: English table and column names and values,
credentials in the request log masked, free text left as is). The loaders need
the German-schema pickle.

Known quirks of the old pickle, tolerated by the self-check: the ``longtext``
columns ``Operationen.OpBericht`` and ``Prozeduren.Befund`` are named
``longtext`` there and keep ``\\r\\n`` escaped. Here they get their real names
and are unescaped.
"""

# ruff: noqa: E501  (long German/English strings in the translation tables)
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus

import pandas as pd

OLD_DUMP = "/data/mice/maus_mysql.sql"
OLD_PICKLE = "/data/mice/maus.pandas.pkl"
NEW_DUMP = "/data/mice/mausprojekt-3.sql"
NEW_PICKLE = "/data/mice/mausprojekt-3.pandas.pkl"
NEW_PICKLE_EN = "/data/mice/mausprojekt-3.en.pandas.pkl"
# Translations of the free text (study data, may name staff: kept next to the data, not in the repository)
FREE_TEXT_EN = "/data/mice/mausprojekt-3.free_text_en.json"

CREATE = re.compile(r"CREATE TABLE `([^`]+)` \((.*?)\n\)[^;]*;", re.S)
COLUMN = re.compile(r"^\s*`([^`]+)`\s+(\w+)", re.M)
INSERT = re.compile(r"INSERT INTO `([^`]+)`\s*(?:\(([^)]*)\))?\s*VALUES")
TOKEN = re.compile(
    r"""\s*(?:
      '((?:[^'\\]|\\.|'')*)'                         # 1 string
    | (NULL)\b                                       # 2
    | b'([01]*)'                                     # 3 bit
    | 0x([0-9A-Fa-f]+)                               # 4 hex
    | (-?\d+)(?![\d.eE])                             # 5 int
    | (-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)       # 6 float
    | ([(),;])                                       # 7 punctuation
    )""",
    re.X | re.S,
)
ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
DATE_TYPES = {"date", "datetime", "timestamp"}
# Long text columns the old pickle names "longtext" and keeps escaped
OLD_LONGTEXT = {"Operationen": "OpBericht", "Prozeduren": "Befund"}


def _unescape(s: str) -> str:
    if "\\" not in s and "''" not in s:
        return s
    return re.sub(r"\\(.)|''", lambda m: ESCAPES.get(m[1], m[1]) if m[1] else "'", s, flags=re.S)


def _read_values(text: str, pos: int) -> tuple[list[list], int]:
    """Rows of ``(...), (...);`` starting at ``pos``, and the position after the ``;``."""
    rows, row = [], None
    while True:
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"Cannot parse the dump at {pos}: {text[pos : pos + 80]!r}")
        pos, kind, value = m.end(), m.lastindex, m[m.lastindex]
        if kind == 7:
            if value == "(":
                row = []
            elif value == ")":
                rows.append(row)
            elif value == ";":
                return rows, pos
        elif kind == 1:
            row.append(_unescape(value))
        elif kind == 2:
            row.append(None)
        elif kind == 3:
            row.append(int(value or "0", 2))
        elif kind == 4:
            row.append(int(value, 16))
        elif kind == 5:
            row.append(int(value))
        else:
            row.append(float(value))


def read_types(path: str) -> dict[str, dict[str, str]]:
    """MySQL column types per table, in ``CREATE TABLE`` order."""
    text = Path(path).read_text(encoding="utf-8")
    return {name: dict(COLUMN.findall(body)) for name, body in CREATE.findall(text)}


def zeitpunkt(visits: pd.DataFrame) -> pd.Series:
    """``Visiten.Zeitpunkt`` of the old pickle: visit date and time, NaT for invalid dates."""
    return pd.to_datetime(visits["Visitendatum"] + " " + visits["Visitenzeit"], errors="coerce")


def read_dump(path: str, like: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """All tables of the MySQL dump at ``path`` (mysqldump or phpMyAdmin format).

    ``like``: cast to the dtypes of these tables, see the module docstring.
    """
    text = Path(path).read_text(encoding="utf-8")
    types = {name: dict(COLUMN.findall(body)) for name, body in CREATE.findall(text)}
    rows = {name: [] for name in types}
    columns = {name: list(cols) for name, cols in types.items()}
    pos = 0
    while m := INSERT.search(text, pos):
        name = m[1]
        insert_columns = re.findall(r"`([^`]+)`", m[2]) if m[2] else columns[name]
        values, pos = _read_values(text, m.end())
        bad = [r for r in values if len(r) != len(insert_columns)]
        if bad:
            raise ValueError(f"{name}: row with {len(bad[0])} values for {len(insert_columns)} columns")
        rows[name].extend(dict(zip(insert_columns, r, strict=True)) for r in values)

    tables = {name: pd.DataFrame(rows[name] or None, columns=columns[name]) for name in sorted(types)}
    return tables if like is None else _cast_like(tables, types, like)


def _cast(column: pd.Series, dtype) -> pd.Series | None:
    """``column`` as ``dtype`` if no value changes, else None."""
    if column.dtype == dtype:
        return column
    if pd.api.types.is_object_dtype(dtype):
        return column.astype(object)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return None
    numbers = pd.to_numeric(column, errors="coerce")
    if numbers.notna().sum() != column.notna().sum():
        return None
    if pd.api.types.is_integer_dtype(dtype) and (numbers.isna().any() or (numbers % 1 != 0).any()):
        return None
    return numbers.astype(dtype)


def _cast_like(tables, types, like):
    out = {}
    for name, frame in tables.items():
        frame = frame.copy()
        reference = like.get(name, pd.DataFrame())
        for column in frame.columns:
            if column in reference.columns:
                cast = _cast(frame[column], reference[column].dtype)
                if cast is not None:
                    frame[column] = cast
            elif types[name][column] in DATE_TYPES:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if "Zeitpunkt" in reference.columns and "Zeitpunkt" not in frame.columns:
            frame["Zeitpunkt"] = zeitpunkt(frame)
        out[name] = frame
    return out


def dtype_differences(tables: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Shared columns whose dtype differs from ``reference``."""
    rows = [
        (name, column, str(reference[name][column].dtype), str(frame[column].dtype))
        for name, frame in tables.items()
        if name in reference
        for column in frame.columns
        if column in reference[name].columns and frame[column].dtype != reference[name][column].dtype
    ]
    return pd.DataFrame(rows, columns=["table", "column", "reference", "dump"])


def self_check(dump: str = OLD_DUMP, pickle: str = OLD_PICKLE) -> None:
    """The old dump must parse to the old pickle, table by table."""
    old = pd.read_pickle(pickle)
    print("dtypes inferred from the dump that differ from the old pickle (cast with like=old):")
    print(dtype_differences(read_dump(dump), old).to_string(index=False))
    parsed = read_dump(dump, like=old)
    assert sorted(parsed) == sorted(old), set(parsed) ^ set(old)
    for name, expected in old.items():
        if name in OLD_LONGTEXT:
            # Tolerated: the old pickle names the long text column "longtext" and keeps \r\n escaped
            real = OLD_LONGTEXT[name]
            expected = expected.rename(columns={"longtext": real})
            expected[real] = expected[real].str.replace("\\r", "\r").str.replace("\\n", "\n")
        pd.testing.assert_frame_equal(parsed[name], expected, check_exact=True, obj=name)
    print(f"Self-check passed: {dump} reproduces {pickle} ({len(old)} tables)")


# ---------------------------------------------------------------------------
# English translation
# ---------------------------------------------------------------------------
# Every table and every column has an English name, and every text column is
# exactly one of: translated value by value (VALUES_EN), free text left as is
# (FREE_TEXT) or language-neutral (NEUTRAL: numbers stored as text, dates,
# codes, URLs, proper nouns). translate() fails on anything unmapped.

REDACTED = "<redacted>"


def redact_log(payload: pd.Series) -> pd.Series:
    """Mask passwords and session ids in the request log (``Logbuch.Bemerkung``).

    Rows start with ``<n> (<user>, <session id or typed password>) - (...)``;
    the second field is masked, as are ``password=``/``sessionid=``/``scode=``
    parameters (also URL-encoded) and any 32-digit hex token.
    """
    s = payload.str.replace(r"^(\S* \([^,()]*, )[^)]*\)", rf"\g<1>{REDACTED})", regex=True)
    s = s.str.replace(r"(?i)((?:^|&)scode=)[^&,)\s]*", rf"\g<1>{REDACTED}", regex=True)
    s = s.str.replace(r"(?i)(password(?:=|%3D))[^&,)\s]*", rf"\g<1>{REDACTED}", regex=True)
    s = s.str.replace(r"(?i)(sessionid(?:=|%3D))[^&,)%\s]*", rf"\g<1>{REDACTED}", regex=True)
    return s.str.replace(r"(?i)\b[0-9a-f]{32}\b", REDACTED, regex=True)


def check_redacted(payload: pd.Series) -> None:
    """No unmasked password, session id or session-like token is left (values are never printed)."""
    s = payload.dropna()
    assert not s.str.contains(rf"(?i)password(?:=|%3D)(?!{REDACTED})[^&,)\s]", regex=True).any(), "password left"
    assert not s.str.contains(rf"(?i)sessionid(?:=|%3D)(?!{REDACTED})[^&,)%\s]", regex=True).any(), "session id left"
    assert not s.str.contains(r"(?i)\b[0-9a-f]{32}\b", regex=True).any(), "hex token left"


def load_free_text(path: str = FREE_TEXT_EN) -> dict[str, dict[str, str]]:
    """Translations of the free text: ``{"Table.Column": {German: English}}`` (original table and column names)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


PAYLOAD = re.compile(r"^(\S* \([^,()]*, [^)]*\)) - \((.*)\)$", re.S)
PAYLOAD_PREFIX_ONLY = re.compile(r"^\S* \([^,()]*, [^)]*\)$")
# Request parameters whose values are user names or credentials: never translated (credentials are masked)
LOG_KEEP_VALUES = {"user", "username", "password", "sessionid", "Sessionid", "sessID", "scode"}


def _translate_value(value: str, mapping: dict[str, str]) -> str:
    """One URL-encoded parameter value: German text is translated and re-encoded, codes, numbers and URLs are kept."""
    text = unquote_plus(value)
    if not re.search(r"[A-Za-zÄÖÜäöüß]{2}", text) or re.match(r"(https?://|\$url\$)", text):
        return value
    assert text in mapping, f"untranslated request parameter value: {text[:60]!r}"
    return value if mapping[text] == text else quote_plus(mapping[text])


def translate_payload(payload: pd.Series, mapping: dict[str, str]) -> tuple[pd.Series, int]:
    """English request log: parameter names via LOG_PARAMS_EN, values via ``mapping``.

    Returns the translated payloads and the number of non-empty payloads that could not be parsed (left as they are).
    """
    cache, failures = {}, 0

    def params(part):
        out = []
        for kv in part.split("&") if part else []:
            name, eq, value = kv.partition("=")
            english = "selrep" + name[6:] if re.fullmatch(r"selrep_\d+", name) else LOG_PARAMS_EN.get(name)
            assert english is not None, f"no English name for request parameter {name!r}"
            out.append(
                english + eq + (value if name in LOG_KEEP_VALUES or not eq else _translate_value(value, mapping))
            )
        return "&".join(out)

    def one(text):
        nonlocal failures
        if text not in cache:
            m = PAYLOAD.match(text)
            if m:
                cache[text] = f"{m[1]} - ({',---,'.join(params(part) for part in m[2].split(',---,'))})"
            else:
                failures += bool(text.strip()) and not PAYLOAD_PREFIX_ONLY.match(text)
                cache[text] = text
        return cache[text]

    return payload.map(lambda t: one(t) if isinstance(t, str) else t), failures


def translate(
    tables: dict[str, pd.DataFrame], free_text: dict[str, dict[str, str]] | None = None
) -> dict[str, pd.DataFrame]:
    """English copy of the German-schema ``tables`` (as in the pickles); asserts that everything is mapped.

    ``free_text`` (:func:`load_free_text`) translates the FREE_TEXT columns, which are otherwise left as they are.
    """
    out = {}
    for name, frame in tables.items():
        assert name in TABLES_EN, f"no English name for table {name}"
        columns = COLUMNS_EN[name]
        missing = [c for c in frame.columns if c not in columns]
        assert not missing, f"no English name for {name}.{missing}"
        frame = frame.copy()
        for column in frame.columns:
            key = (name, column)
            kinds = [key in VALUES_EN, key in FREE_TEXT, key in NEUTRAL or key in CODE_PREFIXES_EN]
            if frame[column].dtype != object:
                assert not any(kinds[:2]), f"{name}.{column} is not text"
                continue
            assert sum(kinds) == 1, f"{name}.{column} must be in exactly one of VALUES_EN, FREE_TEXT, NEUTRAL"
            if key in VALUES_EN:
                mapping = VALUES_EN[key]
                values = frame[column].dropna()
                values = values[values.astype(str).str.strip() != ""]
                unmapped = sorted({v for v in values.unique() if v not in mapping and str(v).strip() not in mapping})
                assert not unmapped, f"untranslated values in {name}.{column}: {unmapped}"
                frame[column] = frame[column].map(
                    lambda v, m=mapping: m.get(v, m.get(v.strip(), v)) if isinstance(v, str) else v
                )
            elif key in NEUTRAL:
                text = frame[column].dropna().astype(str)
                assert not text.str.contains("[äöüÄÖÜß]").any() or key in NEUTRAL_PROPER_NOUNS, f"German text in {key}"
            elif key in CODE_PREFIXES_EN:
                frame[column] = frame[column].map(lambda v, k=key: _translate_code(v, CODE_PREFIXES_EN[k]))
            elif free_text is not None and name != "Logbuch":
                mapping = free_text[f"{name}.{column}"]
                values = frame[column].dropna()
                values = values[values.astype(str).str.strip() != ""]
                unmapped = sorted(set(values) - set(mapping))
                assert not unmapped, f"{len(unmapped)} untranslated free-text values in {name}.{column}"
                frame[column] = frame[column].map(lambda v, m=mapping: m.get(v, v) if isinstance(v, str) else v)
        if name == "Logbuch":
            frame["Bemerkung"] = redact_log(frame["Bemerkung"])
            check_redacted(frame["Bemerkung"])
            if free_text is not None:
                frame["Bemerkung"], failures = translate_payload(frame["Bemerkung"], free_text["Logbuch.Bemerkung"])
                check_redacted(frame["Bemerkung"])
                print(f"Request log: {failures} non-empty payloads could not be parsed and are left as they are")
        out[TABLES_EN[name]] = frame.rename(columns=columns)
    return out


def _translate_code(value, prefixes: dict[str, str]):
    """A code ``<German prefix><running number>``, e.g. ``leber_rgb-7-1`` -> ``liver_rgb-7-1``."""
    if not isinstance(value, str) or not re.search("[A-Za-z]", value):
        return value
    prefix = re.sub(r"[-\d]+$", "", value)
    assert prefix in prefixes, f"unknown code prefix {prefix!r}"
    return prefixes[prefix] + value[len(prefix) :]


def free_text_columns(
    tables_en: dict[str, pd.DataFrame], german: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """The free-text columns of the English ``tables_en``: non-empty rows and (from ``german``) unique originals."""
    rows = []
    for name, column in sorted(FREE_TEXT):
        table, column_en = TABLES_EN[name], COLUMNS_EN[name][column]
        if table in tables_en:
            s = tables_en[table][column_en].dropna().astype(str)
            row = {"table": table, "column": column_en, "rows with text": int((s.str.strip() != "").sum())}
            if german is not None:
                g = german[name][column].dropna().astype(str)
                row["unique originals"] = g[g.str.strip() != ""].nunique()
            rows.append(row | {"content": FREE_TEXT[(name, column)]})
    return pd.DataFrame(rows)


TABLES_EN = {
    "Applikation": "routes",
    "Bestrahlungen": "radiotherapy",
    "Blut": "blood_samples",
    "ChTKonzepte": "chemo_protocols",
    "ChemoMed": "chemo_drugs",
    "Chemotherapie": "chemotherapy",
    "Destinationen": "app_destinations",
    "Druckmessungen": "pressure_measurements",
    "Faerbungen": "stainings",
    "Farben": "colours",
    "Gewebe": "tissue_samples",
    "KlinikFirma": "institutions",
    "Knochenmark": "bone_marrow_samples",
    "Kohorten": "cohorts",
    "Kryo": "cryo_samples",
    "Logbuch": "app_log",
    "MRTKontrast": "mri_contrast_agents",
    "MRTSequenzen": "mri_sequences",
    "MRTWichtungen": "mri_weightings",
    "Markierungen": "marking_types",
    "Messkammern": "ionisation_chambers",
    "Module": "app_modules",
    "Narkosen": "anaesthesia",
    "OTBoxTypen": "slide_box_types",
    "OTBoxen": "slide_boxes",
    "OTSchnittKonzept": "sectioning_protocols",
    "OTSchnittProzeduren": "sectioning_procedures",
    "Objekttraeger": "slides",
    "Operationen": "surgeries",
    "Organe": "organs",
    "Organsysteme": "organ_systems",
    "PCR": "pcr",
    "PCR_Import": "pcr_import",
    "Paraffin": "paraffin_samples",
    "Prozeduren": "procedures",
    "Qualifikationen": "qualifications",
    "RTKonzepte": "rt_protocols",
    "Rollen": "roles",
    "SeitenBez": "body_sides",
    "Standorte": "locations",
    "SystemVariablen": "app_settings",
    "Tagescheck": "daily_device_checks",
    "Tiertypen": "strains",
    "Tierversuche": "projects",
    "Todesursachen": "death_causes",
    "Toetungsarten": "kill_methods",
    "Topographie": "sites",
    "Traeger": "drug_vehicles",
    "Tumor": "tumours",
    "Tumorzelllinien": "cell_lines",
    "VGruppe": "injection_batches",
    "Versuchsreihen": "trails",
    "Versuchstiere": "mice",
    "Visiten": "visits",
    "Zuechter": "breeders",
}

_SAMPLE = {
    "IdTier": "mouse_id",
    "Entnahme": "sampling_date",
    "LfdNr": "running_number",
    "LfdNrTier": "running_number_mouse",
    "Jahr": "year",
    "Bemerkung": "remark",
    "Auswertung": "analysed",
    "Type": "type",
}
_PCR = {
    "IdPCR": "pcr_id",
    "IdTier": "mouse_id",
    "IdGewKmBlut": "sample_id",
    "GewebeArt": "sample_type",
    "Datum": "date",
    "Zellzahl1": "cell_count1",
    "Zellzahl2": "cell_count2",
    "Zellzahl3": "cell_count3",
    "MwZellen": "cell_count_mean",
    "StAZellen": "cell_count_sd",
    "Cp1": "cp1",
    "Cp2": "cp2",
    "Cp3": "cp3",
    "MwCp": "cp_mean",
    "StACp": "cp_sd",
    "Bemerkung": "remark",
    "LfdNr": "running_number",
    "LfdNrTier": "running_number_mouse",
    "Jahr": "year",
    "Sonderprobe": "special_sample",
}
_VISIT_ITEMS = {
    "AZFell": "cond_fur",
    "AZKoerperoeffnungen": "cond_body_orifices",
    "AZAugen": "cond_eyes",
    "AZHaltung": "cond_posture",
    "AZMuskeltonus": "cond_muscle_tone",
    "AZHydratation": "cond_hydration",
    "AZKraempfe": "cond_cramps",
    "AZLaehmungen": "cond_paralysis",
    "AZTemperatur": "cond_temperature",
    "AZAtmung": "cond_breathing",
    "VHVerhalten": "beh_behaviour",
    "VHSchlaf": "beh_sleep",
    "VHAnblasen": "beh_reaction_to_blowing",
    "VHBeruehrung": "beh_reaction_to_touch",
    "VHNeugier": "beh_curiosity",
    "VHSozialK": "beh_social_contact",
    "VHMotorik": "beh_motor_function",
    "VHIsolation": "beh_isolation",
    "VHSchmerz": "beh_pain",
    "VHApathie": "beh_apathy",
    "VHKoordination": "beh_coordination",
    "VHAutomutilation": "beh_self_mutilation",
}
COLUMNS_EN = {
    "Applikation": {"IdApplikation": "route_id", "NameApplikation": "name", "Bemerkung": "remark"},
    "Bestrahlungen": {
        "IdBestrahlung": "fraction_id",
        "IdTier": "mouse_id",
        "IdTumor": "tumour_id",
        "IdRTKonzept": "rt_protocol_id",
        "PTVIst": "target_volume",
        "DosisSoll": "planned_dose_gy",
        "DosisIst": "delivered_dose_gy",
        "Monitorunits": "monitor_units",
        "QualitaetIst": "beam",
        "FeldX": "field_x",
        "FeldY": "field_y",
        "Abstand": "distance",
        "EinschubNr": "slot_number",
        "Datum": "date",
        "Beginn": "start_time",
        "Ende": "end_time",
        "Bemerkung": "remark",
        "Unterschrift": "signature",
    },
    "Blut": {"IdBlut": "blood_sample_id", **_SAMPLE},
    "ChTKonzepte": {
        "IdChTKonzept": "chemo_protocol_id",
        "NameChTKonzept": "name",
        **{
            f"{g}{i}": f"{e}{i}"
            for i in (1, 2, 3)
            for g, e in [
                ("IdChTMed", "drug_id_"),
                ("ChTKonz", "dose_mg_per_kg_"),
                ("IdChTApp", "route_id_"),
                ("ChTTage", "days_"),
            ]
        },
        "Bemerkung": "remark",
    },
    "ChemoMed": {
        "IdChemoMed": "drug_id",
        "NameMed": "name",
        "IdTraeger": "vehicle_id",
        "IdKlinikFirma": "institution_id",
        "IdPerson": "person_id",
        "Bemerkung": "remark",
        "Kurz": "short_name",
    },
    "Chemotherapie": {
        "IdChemo": "administration_id",
        "IdChTKonzept": "chemo_protocol_id",
        "IdTier": "mouse_id",
        "IdTumor": "tumour_id",
        "ChTDatum": "date",
        "IdChTMed": "drug_id",
        "ChTKonzIst": "delivered_dose_mg_per_kg",
        "ChTVolumenAppl": "volume_administered",
        "IdApplikationIst": "route_id",
        "ChTBeginn": "start_time",
        "ChTEnde": "end_time",
        "Bemerkung": "remark",
        "Unterschrift": "signature",
    },
    "Destinationen": {"Destination": "destination", "Link": "link", "IdDestination": "destination_id"},
    "Druckmessungen": {
        "Id": "measurement_id",
        "IdTier": "mouse_id",
        "Messdatum": "date",
        "IdTopographie": "site_id",
        "Messwert": "value",
        "Einheit": "unit",
    },
    "Faerbungen": {
        "IdFaerbung": "staining_id",
        "NameFaerbung": "name",
        "KurzNameFaerbung": "short_name",
        "Bemerkung": "remark",
    },
    "Farben": {"IdFarben": "colour_id", "NameFarbe": "name", "Bemerkung": "remark"},
    "Gewebe": {
        "IdGewebe": "tissue_sample_id",
        "IdTier": "mouse_id",
        "IdTumor": "tumour_id",
        "IdTopographie": "site_id",
        "IdSeite": "side_id",
        "Gewebeart": "tissue_type",
        "Entnahme": "sampling_date",
        "Gewicht": "weight_g",
        "Bemerkung": "remark",
        "Auswertung": "analysed",
    },
    "KlinikFirma": {
        "IdKlinikFirma": "institution_id",
        "KlinikFirma": "name",
        "Zentrum": "centre",
        "Abteilung": "department",
        "Strasse": "street",
        "Hausnummer": "house_number",
        "PLZ": "postcode",
        "Ort": "city",
        "Land": "country",
        "Bemerkung": "remark",
    },
    "Knochenmark": {"IdKnochenmark": "bone_marrow_sample_id", **_SAMPLE},
    "Kohorten": {
        "IdKohorte": "cohort_id",
        "KohorteName": "name",
        "Versuchsreihe": "trail_id",
        "NameZelllinie": "cell_line_id",
        "Tiertyp": "strain_id",
        "Bemerkung": "remark",
    },
    "Kryo": {
        "IdKryo": "cryo_sample_id",
        "IdTier": "mouse_id",
        "IdTumor": "tumour_id",
        "IdGewebe": "tissue_sample_id",
        "Probenart": "sample_type",
        "Tonne": "container",
        "Turm": "tower",
        "Schublade": "drawer",
        "Reihe": "row",
        "Spalte": "column",
        "IdFarbe": "colour_id",
        "LfdNr": "running_number",
        "LfdNrTier": "running_number_mouse",
        "Jahr": "year",
        "Bemerkung": "remark",
    },
    "Logbuch": {
        "IdLog": "log_id",
        "IdNummer": "user_number",
        "Ip": "ip",
        "Aktion": "action",
        "Bemerkung": "payload",
        "Zeitstempel": "timestamp",
    },
    "MRTKontrast": {
        "IdMRTKontrast": "contrast_id",
        "NameKontrastmittel": "name",
        "IdKlinikFirma": "institution_id",
        "Bemerkung": "remark",
    },
    "MRTSequenzen": {
        "IdMRTSequenz": "sequence_id",
        "IdTier": "mouse_id",
        "MRTScanner": "scanner",
        "DatumZeit": "datetime",
        "IdMRTWichtung": "weighting_id",
        "NameSequenz": "name",
        "IdMRTKontrast": "contrast_id",
        "Bemerkung": "remark",
    },
    "MRTWichtungen": {"IdWichtung": "weighting_id", "NameWichtung": "name", "Bemerkung": "remark"},
    "Markierungen": {
        "IdMarkierung": "marking_id",
        "Markierungstyp": "name",
        "Bemerkung": "remark",
        "Kurz": "short_name",
        "Seiteninvariant": "side_invariant",
    },
    "Messkammern": {
        "IdMesskammer": "chamber_id",
        "BezMesskammer": "name",
        "NrMesskammer": "serial_number",
        "IdKlinikFirma": "institution_id",
        "DatumKal": "calibration_date",
        "NDW": "calibration_factor",
        "Bemerkung": "remark",
    },
    "Module": {
        "Modulname": "name",
        "Modullink": "link",
        "Shortcut": "shortcut",
        "Platz": "position",
        "Parent": "parent",
        "ModulParent": "parent_module",
        "Icon": "icon",
        "IconGr": "icon_size",
        "Color": "colour",
        "IdModul": "module_id",
    },
    "Narkosen": {
        "IdNarkose": "anaesthesia_id",
        "IdTier": "mouse_id",
        "Narkotikum": "agent",
        "FlowLuft": "flow_air",
        "FlowO2": "flow_o2",
        "FlowN2": "flow_n2",
        "NarkoseDatum": "date",
        "NarkoseBeginn": "start_time",
        "NarkoseEnde": "end_time",
        "Bemerkung": "remark",
    },
    "OTBoxTypen": {
        "IdOTBoxTyp": "box_type_id",
        "NameOTBox": "name",
        "Spalten": "columns",
        "SliceProSpalte": "slots_per_column",
        "OTproSlice": "slides_per_slot",
        "Bemerkung": "remark",
    },
    "OTBoxen": {
        "IdOTBox": "box_id",
        "OTBoxTyp": "box_type_id",
        "Zellen": "cells",
        "Gewebe": "tissue",
        "LfdNr": "running_number",
        "Versuchsreihe": "trail_id",
        "Bemerkung": "remark",
    },
    "OTSchnittKonzept": {
        "IdSchnittkonzept": "sectioning_protocol_id",
        "NameKonzept": "name",
        "Prozeduren": "procedures",
        "Bemerkung": "remark",
    },
    "OTSchnittProzeduren": {
        "IdSchnittprozedur": "sectioning_procedure_id",
        "NameSchnittprozedur": "name",
        "AnzahlSchnitte": "n_sections",
        "Schnitte": "sections",
        "Faerbung": "staining_id",
        "Bemerkung": "remark",
    },
    "Objekttraeger": {
        "IdObjekttraeger": "slide_id",
        "IdTier": "mouse_id",
        "SchnittNr": "section_number",
        "IdParaffin": "paraffin_sample_id",
        "IdFaerbung": "staining_id",
        "AnzMet": "n_metastases",
        "AnzRGB": "n_rgb_cells",
        "Bemerkung": "remark",
        "IndexOTOrt": "storage_index",
    },
    "Operationen": {
        "IdOp": "surgery_id",
        "IdTier": "mouse_id",
        "DatumOP": "date",
        "BeginnOp": "start_time",
        "EndeOp": "end_time",
        "Analgetikum": "analgesic",
        "OpBericht": "report",
        "Operateur": "surgeon_id",
        "Tumorgewicht": "tumour_weight_g",
    },
    "Organe": {"IdOrgan": "organ_id", "NameOrgan": "name", "AbkOrgan": "abbreviation", "Bemerkung": "remark"},
    "Organsysteme": {"IdOrgansystem": "organ_system_id", "NameOrgansystem": "name", "Bemerkung": "remark"},
    "PCR": _PCR,
    "PCR_Import": {k: v for k, v in _PCR.items() if k not in ("Zellzahl3", "Cp3")},
    "Paraffin": {
        "IdParaffin": "paraffin_sample_id",
        "IdTier": "mouse_id",
        "IdTumor": "tumour_id",
        "IdGewebe": "tissue_sample_id",
        "Probenart": "sample_type",
        "Histokasten": "cassette",
        "Tonne": "container",
        "Schublade": "drawer",
        "LfdNr": "running_number",
        "LfdNrTier": "running_number_mouse",
        "Jahr": "year",
        "Bemerkung": "remark",
    },
    "Prozeduren": {
        "IdProzedur": "procedure_id",
        "IdTier": "mouse_id",
        "Befund": "finding",
        "Prozedur": "procedure",
        "Datum": "date",
        "Uhrzeit": "time",
        "IdPerson": "person_id",
    },
    "Qualifikationen": {"IdQualifikation": "qualification_id", "Qualifikation": "name", "Bemerkung": "remark"},
    "RTKonzepte": {
        "IdRTKonzept": "rt_protocol_id",
        "RTKonzept": "name",
        "PTV": "target_volume",
        "Qualitaet": "beam",
        "Einzeldosis": "dose_per_fraction_gy",
        "Enddosis": "total_dose_gy",
        "Fraktionen": "fractions",
        "FrakTag": "fractions_per_day",
        "FrakWo": "fractions_per_week",
        "Bemerkung": "remark",
    },
    "Rollen": {"IdRolle": "role_id", "RolleName": "abbreviation", "Bemerkung": "description"},
    "SeitenBez": {"IdSeite": "side_id", "Seite": "name"},
    "Standorte": {
        "IdStandort": "location_id",
        "NameStandort": "name",
        "Raum": "room",
        "Rack": "rack",
        "Bemerkung": "remark",
    },
    "SystemVariablen": {"name": "name", "value": "value", "IdVariable": "setting_id"},
    "Tagescheck": {
        "IdCheck": "check_id",
        "Datum": "date",
        "Zeit": "time",
        "IdPerson": "person_id",
        "Einfahrmodus": "mode",
        "IdMesskammer": "chamber_id",
        "Temperatur": "temperature",
        "Luftdruck": "air_pressure",
        "Dosiswert1": "dose_reading1",
        "Dosiswert2": "dose_reading2",
        "LichtfeldOK": "light_field_ok",
        "LaserOK": "laser_ok",
        "Fehler": "error",
        "Bemerkung": "remark",
    },
    "Tiertypen": {
        "IdTiertyp": "strain_id",
        "Tierart": "species",
        "Tiertyp": "name",
        "Sicherheitsstufe": "safety_level",
        "ZSpender": "cell_donor",
        "ZEmpfaenger": "cell_recipient",
        "Bemerkung": "remark",
    },
    "Tierversuche": {
        "IdVersuch": "project_id",
        "Projekttitel": "title",
        "Versuchsnummer": "licence",
        "Projektleiter": "leader_id",
        "SVProjektleiter": "deputy_leader_id",
        **{f"Ansprechpartner{i}": f"contact{i}_id" for i in range(1, 6)},
        "Bemerkung": "remark",
    },
    "Todesursachen": {"IdTodesursache": "death_cause_id", "Todesursache": "name", "Bemerkung": "remark"},
    "Toetungsarten": {"IdToetungsart": "kill_method_id", "Toetungsart": "name", "Bemerkung": "remark"},
    "Topographie": {
        "IdTopographie": "site_id",
        "IdOrgansystem": "organ_system_id",
        "IdOrgan": "organ_id",
        "Bemerkung": "description",
    },
    "Traeger": {"IdTraeger": "vehicle_id", "NameTraeger": "name", "Bemerkung": "remark"},
    "Tumor": {
        "IdTumor": "tumour_id",
        "IdTier": "mouse_id",
        "IdTopographie": "site_id",
        "IdSeite": "side_id",
        "IdZelllinie": "cell_line_id",
        "TransDatum": "injection_date",
        "EntnahmeDatum": "removal_date",
        "TuGewicht": "tumour_weight_g",
        "Bemerkung": "remark",
    },
    "Tumorzelllinien": {
        "IdZelllinie": "cell_line_id",
        "NameZelllinie": "name",
        "IdTiertyp": "strain_id",
        "Sicherheitsstufe": "safety_level",
        "RGB": "rgb_marked",
        "Bemerkung": "remark",
    },
    "VGruppe": {
        "IdVGruppe": "batch_id",
        "Kohorte": "cohort_id",
        "Zellen": "cells",
        "Zelllinie": "cell_line_id",
        "Datum": "injection_date",
        "Bemerkung": "remark",
    },
    "Versuchsreihen": {
        "IdVersuchsreihe": "trail_id",
        "NameReihe": "name",
        "Versuchsziel1": "aim1",
        "Versuchsziel2": "aim2",
        "Versuchsziel3": "aim3",
        "IdVersuch": "project_id",
        "VersuchStart": "start",
        "VersuchEnde": "end",
        "aktiv": "active",
        **{f"VPartner{i}": f"partner{i}_id" for i in range(1, 6)},
        "Bemerkung": "remark",
    },
    "Versuchstiere": {
        "IdTier": "mouse_id",
        "IdTBase": "tbase_id",
        "IdKurz": "short_id",
        "Versuchsreihe": "trail_id",
        "Vater": "father",
        "Mutter": "mother",
        "Geburtsdatum": "birth_date",
        "Geschlecht": "sex",
        **{f"Markierung{i}{s}": f"marking{i}_{side}" for i in (1, 2) for s, side in (("L", "left"), ("R", "right"))},
        **{f"AnzM{i}{s}": f"n_marking{i}_{side}" for i in (1, 2) for s, side in (("L", "left"), ("R", "right"))},
        "Sterbedatum": "death_date",
        "Todesursache": "death_cause_id",
        "Toetungsart": "kill_method_id",
        "AktivV": "active",
        "Zuechter": "breeder_id",
        "IdTiertyp": "strain_id",
        "RTKonzept": "rt_protocol_id",
        "ChTKonzept": "chemo_protocol_id",
        "Standort": "location_id",
        "Kaefignummer": "cage_number",
        "Rolle": "role_id",
        "VGruppe": "batch_id",
        "Bemerkung": "remark",
    },
    "Visiten": {
        "IdVisite": "visit_id",
        "IdTier": "mouse_id",
        "Gewicht": "weight_g",
        "KGVerlauf": "rel_weight_pct",
        "KGPunkte": "weight_points",
        **_VISIT_ITEMS,
        "AZPunkte": "condition_points",
        "VHPunkte": "behaviour_points",
        "TUVolumen": "tumour_volume",
        "TUExulceration": "tumour_ulceration",
        "TUPunkte": "tumour_points",
        "Score": "score",
        "Belastung": "burden_note",
        "Bemerkung": "remark",
        "Visitendatum": "date",
        "Visitenzeit": "time",
        "TUVolumen2": "tumour2_volume",
        "TUExulceration2": "tumour2_ulceration",
        "TUPunkte2": "tumour2_points",
        "Zeitpunkt": "timestamp",
    },
    "Zuechter": {
        "IdZuechter": "breeder_id",
        "IdKlinikFirma": "institution_id",
        **{f"KontaktZ{i}": f"contact{i}_id" for i in (1, 2, 3)},
        "Bemerkung": "remark",
    },
}


def _same(*values):
    return {v: v for v in values}


_YES = {"ja": "yes"}
_SAFETY = {"S1": "S1", "keine": "none"}
_LUC = "PC-3 RGB + luciferase"
_LUC_DR = "PC-3DR RGB + luciferase"
_COLLAB = {
    "Kooperation Zellmigration": "Collaboration cell migration",
    "Kooperation Olaparib": "Collaboration olaparib",
    "Kooperation Mammabestrahlung": "Collaboration breast irradiation",
    "Kooperation Brustkrebs": "Collaboration breast cancer",
    "Kooperation": "Collaboration",
}
_PROJECT_TITLES = {
    "Der Einfluss von Bestrahlung, Chemotherapie, simultaner Radiochemotherapie und Operation auf die Metastasierung "
    "in Xenograftmausmodellen": "Influence of radiotherapy, chemotherapy, simultaneous radiochemotherapy and surgery on metastasis in "
    "xenograft mouse models",
    "Untersuchung der selektinabhängigen Interaktion verschiedener Zellpopulationen im klinisch relevanten humanen "
    "Tumormodell in der Maus unter Therapie": "Selectin-dependent interaction of cell populations in a clinically relevant human tumour model in the mouse "
    "under therapy",
    "Resistenz und Knochenmarkmikrometastasierung bei antiandrogener Therapie des Mammakarzinoms": "Resistance and bone-marrow micrometastasis under anti-androgen therapy of breast cancer",
    "Bedeutung der TAM-Rezeptoren-Familie bei myeloischen Leukämien": "Role of the TAM receptor family in myeloid leukaemias",
    "Vorversuch 2 Tiere - Methoden lernen": "Pilot with 2 animals - learning methods",
    "Untersuchung der Tumor-spezifischen Strahlensensibilisierung mittels gerichteter Inhibierung der DNA-Reparatur": "Tumour-specific radiosensitisation by targeted inhibition of DNA repair",
    "Strahlensensibilisierung PTEN- bzw. CHD1-depletierter Prostatakarzinome durch PARP1-Inhibition": "Radiosensitisation of PTEN- or CHD1-depleted prostate carcinomas by PARP1 inhibition",
    "TAS102": "TAS102",
    "Der Einfluss von lokal ablativer Strahlenbehandlung, Chemotherapie, Radio-Chemotherapie und Operation auf die "
    "Metastasierung in zwei Xenograft Maus-Prostata-Modellen": "Influence of locally ablative radiotherapy, chemotherapy, radiochemotherapy and surgery on metastasis in two "
    "xenograft mouse prostate models",
    "Immunmodulation durch Strahlentherapie bei Glioblastomen": "Immunomodulation by radiotherapy in glioblastoma",
    "Relevanz klonaler Heterogenität für den cerebralen Tropismus und das strahlentherapeutische Ansprechen von "
    "metastatischen und primären Hirntumoren": "Relevance of clonal heterogeneity for the cerebral tropism and radiotherapy response of metastatic and "
    "primary brain tumours",
    "Nachverfolgung von Systemtherapeutika, Nanopartikeln und Immunzellen in einem Xenograft Maus-Prostata-Tumormodell "
    "mit Röntgenfluoreszenz-Bildgebung": "Tracking of systemic drugs, nanoparticles and immune cells in a xenograft mouse prostate tumour model with "
    "X-ray fluorescence imaging",
    "Darstellung der funktionellen Rolle von small RNA Clustern in der Hepatokarzinogenese": "Functional role of small-RNA clusters in hepatocarcinogenesis",
    'Organentnahmeantrag als Ergänzung zum Projekt "Nachverfolgung von Systemtherapeutika, Nanopartikeln un '
    'Immunzellen in einem Xenograft Maus-Prostata-Tumormodell mit Röntgenfluoreszenzbildgebung" (N029/2023)': 'Organ-removal application supplementing project "Tracking of systemic drugs, nanoparticles and immune cells '
    'in a xenograft mouse prostate tumour model with X-ray fluorescence imaging" (N029/2023)',
}
_ORGANS = {
    "Primärtumor": "primary tumour",
    "Cerebrum": "cerebrum",
    "Cerebellum": "cerebellum",
    "Myelon": "spinal cord",
    "Pulmo": "lung",
    "Cor": "heart",
    "Thymus": "thymus",
    "LK Kopf / Hals": "lymph nodes head / neck",
    "LK Thorax": "lymph nodes thorax",
    "LK Abdomen": "lymph nodes abdomen",
    "LK Extremitäten": "lymph nodes limbs",
    "Mundhöhle": "oral cavity",
    "Oesophagus": "oesophagus",
    "Gaster": "stomach",
    "Darm": "intestine",
    "Peritoneum": "peritoneum",
    "Anus": "anus",
    "Hepar": "liver",
    "Ren": "kidney",
    "Gl. suprarenalis": "adrenal gland",
    "Lien": "spleen",
    "KM Femur": "bone marrow femur",
    "KM Tibia": "bone marrow tibia",
    "Femur": "femur",
    "Tibia": "tibia",
    "Humerus": "humerus",
    "Cutis": "skin",
    "Musculus": "muscle",
    "Ovar": "ovary",
    "Uterus": "uterus",
    "Testis": "testis",
    "Vesica urinaria": "urinary bladder",
    "KM rechtes Bein": "bone marrow right leg",
}
_ORGAN_REMARKS = {
    "Großhirn": "cerebrum",
    "Kleinhirn": "cerebellum",
    "Rückenmark": "spinal cord",
    "Lunge": "lung",
    "Herz": "heart",
    "Thymus": "thymus",
    "Lymphknoten": "lymph nodes",
    "Mundhöhle": "oral cavity",
    "Speiseröhre": "oesophagus",
    "Magen": "stomach",
    "Darm": "intestine",
    "Bauchfell": "peritoneum",
    "After": "anus",
    "Leber": "liver",
    "Niere": "kidney",
    "Nebenniere": "adrenal gland",
    "Milz": "spleen",
    "Knochenmark": "bone marrow",
    "Oberschenkel": "thigh (femur)",
    "Schienenbein": "shin (tibia)",
    "Oberarm": "upper arm (humerus)",
    "Haut": "skin",
    "Muskel": "muscle",
    "Eierstock": "ovary",
    "Gebärmutter": "uterus",
    "Hoden": "testis",
    "Harnblase": "urinary bladder",
    "KM Femur und Tibia rechtes Bein": "bone marrow femur and tibia, right leg",
}
_MODULES = {
    "Home": "Home",
    "Logout": "Logout",
    "Meine Daten": "My data",
    "Admin": "Admin",
    "Personen": "People",
    "Chemotherapie-Konzepte": "Chemotherapy protocols",
    "Institute und Firmen": "Institutes and companies",
    "Tierversuche": "Animal-experiment projects",
    "Tiere": "Animals",
    "Topographie": "Sites",
    "Verfahren": "Methods",
    "Strahlentherapie-Konzepte": "Radiotherapy protocols",
    "Zellen/Histo": "Cells/histology",
    "Logbuch": "Log",
    "Server": "Server",
    "Chemotherapie": "Chemotherapy",
    "Strahlentherapie": "Radiotherapy",
    "Radio-Chemotherapie": "Radiochemotherapy",
    "Diagnostik": "Diagnostics",
    "Tumorzellimplantation": "Tumour cell implantation",
    "Versuchstier Basisdaten": "Animal base data",
    "Tiervisiten": "Animal visits",
    "Neue Applikation": "New administration",
    "Neue Fraktion": "New fraction",
    "Tagescheck": "Daily check",
    "DB Backup": "DB backup",
    "Freitext-Befunde": "Free-text findings",
    "Konzept-Zuordnung": "Protocol assignment",
    "Käfigkarten": "Cage cards",
    "Auswertung": "Analysis",
    "Report generieren": "Generate report",
    "Sektion": "Necropsy",
    "Tod ohne Sektion": "Death without necropsy",
    "Neue vollständige Sektion": "New complete necropsy",
    "Lagerorte RGB ergänzen": "Add RGB storage locations",
    "Gewebeübersicht": "Tissue overview",
    "Nachweis Laborbuch": "Lab book record",
    "Wachstumstabelle": "Growth table",
    "Arbeitslisten": "Work lists",
    "Dokumentation": "Documentation",
    "Histokästen": "Histology cassettes",
    "Objektträger": "Slides",
    "MRT": "MRI",
    "PCR": "PCR",
    "Chemorechner": "Chemo calculator",
    "Tools": "Tools",
    "Stapelliste": "Batch list",
    "Bestrahlungsrechner": "Irradiation calculator",
    "QR-Codes": "QR codes",
    "Tierrelationen": "Animal relations",
    "Status PCR": "PCR status",
    "Nachweise importieren": "Import records",
    "Versuchsarme auswerten": "Analyse experiment arms",
    "Operative Eingriffe": "Surgical procedures",
    "Operationsbericht hinzuf&uuml;gen": "Add surgery report",
    "Operationsnachweise drucken": "Print surgery records",
    "Stapelliste Auswahl": "Batch list selection",
}

VALUES_EN = {
    ("Applikation", "NameApplikation"): {
        "unbekannt": "unknown",
        "Schlundsonde": "oral gavage",
        **_same("i.p.", "i.a.", "i.v.", "s.c."),
    },
    ("Bestrahlungen", "PTVIst"): {
        "Tumor": "tumour",
        "Femur re.": "right femur",
        "Femur li.": "left femur",
        "Tumor Mamma li.": "tumour, left mammary gland",
    },
    ("Blut", "Auswertung"): _YES,
    ("Knochenmark", "Auswertung"): _YES,
    ("Gewebe", "Auswertung"): _YES,
    ("ChTKonzepte", "NameChTKonzept"): {
        "BC Standard alt": "BC standard (old)",
        "BC Standard neu": "BC standard (new)",
        "Cisplatin Dosis 1/3": "Cisplatin dose 1/3",
        "Cisplatin Dosis 2/3": "Cisplatin dose 2/3",
        "Etoposid Dosis 1/3": "Etoposide dose 1/3",
        "Etoposid Dosis 2/3": "Etoposide dose 2/3",
        "Etoposid Dosis 3/3": "Etoposide dose 3/3",
        "Cisplatin + Pause + Etoposid": "Cisplatin + pause + etoposide",
        "Cisplatin red. + Pause + Etoposid": "Cisplatin reduced + pause + etoposide",
        "Olaparib 2h vor RT": "Olaparib 2 h before RT",
        "Ruthenium-Glutamin d1": "Ruthenium glutamine d1",
        **_same(
            "Doctaxel 20 mg/kg d1-8-15",
            "Cisplatin 6mg/kg d1",
            "TAS102 d1-10",
            "Capecitabine d1-5",
            "TAS102 d1-5",
            "Capecitabine d1-10",
            "Docetaxel d1,8 4 mg/kg",
            "Docetaxel D1 8mg/kg",
        ),
    },
    ("ChTKonzepte", "Bemerkung"): {
        "Cisplatin d1, Etoposid d3-5": "Cisplatin d1, etoposide d3-5",
        "Cisplatin 6 mg/kg KG d1": "Cisplatin 6 mg/kg body weight d1",
        "TAS102 150mg/kg KG d1-10": "TAS102 150 mg/kg body weight d1-10",
        "TAS102 150mg/kg KG d1-5": "TAS102 150 mg/kg body weight d1-5",
        "Capecitabine 400 mg/kg KG d1-10": "Capecitabine 400 mg/kg body weight d1-10",
        "Ruthenium-Glutamin d1": "Ruthenium glutamine d1",
        **_same("Capecitabine d1-5", "Docetaxel d1,8 4mg/kg"),
    },
    ("ChemoMed", "NameMed"): (
        _drugs := {
            "Etoposid": "Etoposide",
            "Ruthenium-Glutamin": "Ruthenium glutamine",
            **_same("Cisplatin", "Docetaxel", "Olaparib", "TAS102", "Capecitabine"),
        }
    ),
    ("ChemoMed", "Kurz"): _drugs,
    ("Faerbungen", "NameFaerbung"): {
        "Hämatoxylin-Eosin-Färbung": "Haematoxylin-eosin staining",
        "Antigen Ki67": "Antigen Ki67",
        "Sphingosin-1-Phosphat-Rezeptor": "Sphingosine-1-phosphate receptor",
        "Feulgen-Färbung": "Feulgen staining",
        "CD45-Rezeptorfärbung": "CD45 receptor staining",
        "CD68-Rezeptorfärbung": "CD68 receptor staining",
        "Nervenzelladhäsionsmolekül": "Neural cell adhesion molecule",
        "Alpha1-CAM": "Alpha1-CAM",
    },
    ("Faerbungen", "Bemerkung"): {
        "Standard": "standard",
        "Bestimmung der Wachstumsfraktion": "determination of the growth fraction",
        "Aktivitätssteigerung zytoskeletaler Bewegungsfunktionen": "increased activity of cytoskeletal motility",
        "Färbung von Zellkernen": "staining of cell nuclei",
        "Leukozyten Tyrosinphosphatase": "leukocyte tyrosine phosphatase",
        "Makrophagen": "macrophages",
        "Immunglobulin-Superfamilie": "immunoglobulin superfamily",
        "Zelladhäsionsmolekül": "cell adhesion molecule",
    },
    ("Farben", "NameFarbe"): {
        "hellblau": "light blue",
        "weiß": "white",
        "gelb": "yellow",
        "orange": "orange",
        "rot": "red",
        "lila": "purple",
        "pink": "pink",
        "grün": "green",
        "braun": "brown",
    },
    ("Gewebe", "Gewebeart"): {
        "lilu": "left lung",
        "relu": "right lung",
        "leber_ganz": "liver, whole",
        "leber_gr_1": "liver, large lobe 1",
        "leber_gr_2": "liver, large lobe 2",
        "leber_gr_3": "liver, large lobe 3",
        "hirn_li": "brain, left",
        "hirn_re": "brain, right",
        "tumor": "tumour",
        "relk": "right lung, small piece",
        "reniere": "right kidney",
        "liniere": "left kidney",
        "tumor Abdomen": "tumour, abdomen",
        "leber_riecken": "liver (Riecken)",
        "leber_gr_1_18": "liver, large lobe 1 (2018 scheme)",
        "leber_gr_2_18": "liver, large lobe 2 (2018 scheme)",
        "leber_gr_3_18": "liver, large lobe 3 (2018 scheme)",
        "spleen": "spleen",
    },
    ("KlinikFirma", "KlinikFirma"): {
        "Universitätsklinikum Hamburg-Eppendorf": "University Medical Center Hamburg-Eppendorf",
        **_same("Charles River Laboratories International, Inc.", "IBA Dosimetry", "Servier", "Janvier"),
    },
    ("KlinikFirma", "Zentrum"): {
        "Ambulanzzentrum des UKE GmbH": "UKE outpatient centre (GmbH)",
        "Zentrum für Experimentelle Medizin": "Center for Experimental Medicine",
        "Dekanat": "Dean's office",
        "USA": "USA",
        "Onkologisches Zentrum": "Oncology Center",
        "Zentrum für Radiologie und Endoskopie": "Center for Radiology and Endoscopy",
        "Kopf- und Neurozentrum": "Head and Neurocenter",
        "Zentrum für Innere Medizin": "Center for Internal Medicine",
    },
    ("KlinikFirma", "Abteilung"): {
        "Bereich Strahlentherapie": "Radiotherapy section",
        "Institut für Anatomie und Experimentelle Morphologie": "Institute of Anatomy and Experimental Morphology",
        "Versuchstierhaltung": "Animal facility",
        "Klinik für Stammzelltransplantation (KMT)": "Department of Stem Cell Transplantation",
        "Klinik und Poliklinik für diagnostische und interventionelle Radiologie": "Department of Diagnostic and Interventional Radiology",
        "Institut für Tumorbiologie": "Institute of Tumour Biology",
        "II. Med. Klinik und Poliklinik": "II. Department of Medicine",
        "Labor für Strahlenbiologie und Radioonkologie": "Laboratory of Radiobiology and Radiation Oncology",
        "Klinik und Poliklinik für Neurochirurgie": "Department of Neurosurgery",
        "Forschungstierhaltung": "Research animal facility",
        "I. Med. Klinik und Poliklinik": "I. Department of Medicine",
    },
    ("KlinikFirma", "Strasse"): _same("Martinistraße", "Ballardvale Street", "Bahnhofstraße", "Martinistrasse"),
    ("KlinikFirma", "Land"): {"Frankreich": "France", "Deutschland": "Germany"},
    ("KlinikFirma", "Bemerkung"): {
        "Kooperationsprojekt": "Collaboration project",
        **_same("Charles River Laboratories", "Servier", "Janvier"),
    },
    ("Kohorten", "KohorteName"): {
        "Vorversuch HT29": "Pilot HT29",
        "Vorversuch OH1": "Pilot OH1",
        "Vorversuch H69": "Pilot H69",
        "Vorversuch LNCaP": "Pilot LNCaP",
        "Vorversuch PC3": "Pilot PC3",
        "Dosisfindung Cp/Eto": "Dose finding cisplatin/etoposide",
        "MRT-Kontrolltiere": "MRI control animals",
        "Kooperation Loges": "Collaboration Loges",
        "Kooperation Brustkrebs PymT": "Collaboration breast cancer PymT",
        "Kooperation Brustkrebs MCF7": "Collaboration breast cancer MCF7",
        "Kooperation Nanopartikel": "Collaboration nanoparticles",
        "Vorversuch H69AR RGB Luziferase": "Pilot H69AR RGB luciferase",
        "Vorversuch Olaparib": "Pilot olaparib",
        "Hauptversuch TAS102 HT29": "Main experiment TAS102 HT29",
        "Hauptversuch TAS102 HCT116": "Main experiment TAS102 HCT116",
        "Hauptversuch TAS102 CACO-2": "Main experiment TAS102 CACO-2",
        "Hauptversuch TAS102 SW48": "Main experiment TAS102 SW48",
        "Kooperation Zellmigration": "Collaboration cell migration",
        "PC-3 RGB + Luziferase Vorversuch": f"{_LUC} pilot",
        "PC-3 RGB + Luziferase Hauptversuch": f"{_LUC} main experiment",
        "GBM Vorversuch": "GBM pilot",
        "GBM Hauptversuch": "GBM main experiment",
        "GBM Maire Hauptversuch": "GBM Maire main experiment",
        "Hauptversuch DESY": "Main experiment DESY",
        "PC-3DR RGB + Luziferase Hauptversuch": f"{_LUC_DR} main experiment",
        "Vorversuch N068/2024": "Pilot N068/2024",
        "Organentnahme für ALU-PCR": "Organ removal for ALU-PCR",
        **_same("OH1 RGB", "H69 RGB", "LNCaP RGB", "PC3 RGB"),
    },
    ("Kohorten", "Bemerkung"): {
        "Test der Tumorzellinjektion": "Test of the tumour cell injection",
        "Vorversuch OH1 RGB": "Pilot OH1 RGB",
        "Vorversuch H69 RGB": "Pilot H69 RGB",
        "Vorversuch LNCaP RGB": "Pilot LNCaP RGB",
        "Vorversuch PC3 RGB": "Pilot PC3 RGB",
        "Dosisfindung für Cisplatin und Etoposid (ohne Tumorzellinjektion)": "Dose finding for cisplatin and etoposide (without tumour cell injection)",
        "Hauptversuch OH1 RGB": "Main experiment OH1 RGB",
        "Hauptversuch H69 RGB": "Main experiment H69 RGB",
        "Hauptversuch LNCaP RGB": "Main experiment LNCaP RGB",
        "Hauptversuch PC3 RGB": "Main experiment PC3 RGB",
        "Tiere ohne Tumorzellen": "Animals without tumour cells",
        "Kooperation Loges": "Collaboration Loges",
        "Hauptversuch PymT": "Main experiment PymT",
        "Hauptversuch MCF7": "Main experiment MCF7",
        "Kooperation Nanopartikel": "Collaboration nanoparticles",
        "Vorversuch Zellmigration Luziferase und RGB": "Pilot cell migration luciferase and RGB",
        "TAS102 Zellen HT29": "TAS102 cells HT29",
        "TAS102 Zellen HCT116": "TAS102 cells HCT116",
        "TAS102 Zellen CACO-2": "TAS102 cells CACO-2",
        "TAS102 Zellen SW48": "TAS102 cells SW48",
        "Vorversuch PC-3 RGB + Luziferase": f"Pilot {_LUC}",
        "Hauptversuch PC-3 RGB + Luziferase": f"Main experiment {_LUC}",
        "Vorversuch N006/2020": "Pilot N006/2020",
        "Hauptversuch N006/2020": "Main experiment N006/2020",
        "Hauptversuch N052/2020": "Main experiment N052/2020",
        "Hauptversuch N029/2023": "Main experiment N029/2023",
        "Hauptversuch PC-3DR RGB + Luziferase": f"Main experiment {_LUC_DR}",
        "Vorversuch N068/2024": "Pilot N068/2024",
        "Organentnahme": "Organ removal",
        **_COLLAB,
    },
    ("Kryo", "Probenart"): {
        "tumor_trizol": "tumour in TRIzol",
        "alupcr_gew": "ALU-PCR tissue",
        "tumor_kryorgb": "tumour cryo RGB",
        "tumor_kryojuecker": "tumour cryo (Jücker)",
        "tumor_kryorgb_18": "tumour cryo RGB (2018 scheme)",
        "spleen_kryo": "spleen cryo",
        "tumor_kryogel": "tumour cryo gel",
        "tumor_kryo": "tumour cryo",
        "alupcr_cryo": "ALU-PCR cryo",
    },
    ("Kryo", "Tonne"): {
        "Stickstoffbad Tür": "liquid-nitrogen tank, door",
        "-80 Grad Keller": "-80 °C freezer, basement",
    },
    ("MRTKontrast", "NameKontrastmittel"): {
        "kein Kontrastmittel": "no contrast agent",
        "mit Std. Kontrastmittel": "with standard contrast agent",
    },
    ("MRTSequenzen", "NameSequenz"): {
        "Diverse Tumortest": "various, tumour test",
        "Diverse mit Kontrastmittel": "various, with contrast agent",
    },
    ("MRTWichtungen", "NameWichtung"): {
        "unbekannt": "unknown",
        "Protonen": "proton density",
        **_same("T1", "T2", "FLAIR"),
    },
    ("Markierungen", "Markierungstyp"): {
        "Keine": "none",
        "Ohrlochung": "ear punch",
        "Pikrinsäure Kopf": "picric acid, head",
        "Pikrinsäure Rumpf": "picric acid, trunk",
        "Pikrinsäure Schwanz": "picric acid, tail",
        "Transponder": "transponder",
        "Pikrinsäure Flanke links": "picric acid, left flank",
        "Tätowierung Vorderlauf": "tattoo, foreleg",
        "Tätowierung Hinterlauf": "tattoo, hind leg",
        "Strich Schwanz": "line on tail",
        "Pikrinsäure Flanke rechts": "picric acid, right flank",
    },
    ("Markierungen", "Kurz"): {
        "Keine": "none",
        "OL": "EP",
        "PS Kopf": "PA head",
        "PS Rumpf": "PA trunk",
        "PS Schwanz": "PA tail",
        "Tp": "Tp",
        "PS Flanke li.": "PA left flank",
        "T VL": "T FL",
        "T HL": "T HL",
        "PS Flanke re.": "PA right flank",
    },
    ("Markierungen", "Seiteninvariant"): _YES,
    ("Messkammern", "BezMesskammer"): {"Monitorkammer": "monitor chamber", **_same("CC13", "CC01")},
    ("Messkammern", "NrMesskammer"): {"unbekannt": "unknown", **_same("5471", "4481")},
    ("Module", "Modulname"): _MODULES,
    ("Module", "Parent"): {"bereiche": "sections"},
    ("Module", "ModulParent"): {
        "meinedaten": "my_data",
        "vt": "animals",
        "doku": "documentation",
        **_same("home", "admin", "chemo", "radio", "report", "dissection", "tools", "op"),
    },
    ("Narkosen", "Narkotikum"): {
        "Isofluran": "Isoflurane",
        "Iso": "Isoflurane (Iso)",
        **_same("CO2+O2", "C02 O2", "CO2 O2", "CO2 + O2", "CO2O2", "Rompun/Ketanest i.p."),
    },
    ("OTBoxTypen", "NameOTBox"): {"Große Box": "large box", "Große Box einfach": "large box, single"},
    ("OTBoxTypen", "Bemerkung"): {
        "Große Histobox mit Doppelschlitz": "large histology box with double slots",
        "Große Box, einfach belegt": "large box, single occupancy",
    },
    ("OTBoxen", "Gewebe"): {"Lunge": "lung"},
    ("OTBoxen", "Bemerkung"): {"Box1 der Lunge": "box 1 of the lung"},
    ("OTSchnittKonzept", "NameKonzept"): {"Lunge": "lung"},
    ("OTSchnittKonzept", "Bemerkung"): {"Lunge Standard": "lung standard"},
    ("OTSchnittProzeduren", "NameSchnittprozedur"): {"LungeSerienschnitt": "lung serial sections"},
    ("OTSchnittProzeduren", "Bemerkung"): {"StandardSchnitte": "standard sections"},
    ("Operationen", "Analgetikum"): _same("Ibuprofen s.c.", "Carprofen s.c."),
    ("Organe", "NameOrgan"): _ORGANS,
    ("Organe", "AbkOrgan"): {"keine": "none", **_same(*"ANRBISXVEFCKDHTUMLOPZ", "NN")},
    ("Organe", "Bemerkung"): _ORGAN_REMARKS,
    ("Organsysteme", "NameOrgansystem"): {
        "Hirn + Nervensystem": "brain + nervous system",
        "Respirationssystem": "respiratory system",
        "Vaskuläres System": "vascular system",
        "Immunsystem": "immune system",
        "HNO-Trakt": "ENT tract",
        "Gastrointestinalsystem": "gastrointestinal system",
        "Knochen": "bone",
        "Haut": "skin",
        "Weichgewebe": "soft tissue",
        "Reproduktionssystem": "reproductive system",
        "Urogenitalsystem": "urogenital system",
        "Hämatopoetisches System": "haematopoietic system",
    },
    ("PCR", "GewebeArt"): {"bm": "bone marrow", **_same("blood", "lung", "brain", "liver", "spleen")},
    ("Paraffin", "Probenart"): {
        "hk_agar_paraffin": "tissue cassette, agar + paraffin",
        "leber_rgb": "liver RGB",
        "hirn_paraffin": "brain paraffin",
        "tumor_paraffin": "tumour paraffin",
        "sonst_paraffin": "other paraffin",
        "leber_paraffin": "liver paraffin",
        "tumor_rgb": "tumour RGB",
        "tumor_Abdomen": "tumour, abdomen",
        "spleen_facs": "spleen FACS",
        "tumor_facs": "tumour FACS",
    },
    ("Qualifikationen", "Qualifikation"): {
        "Arzt": "physician",
        "BTA": "biological technical assistant",
        "Naturwissenschaftler": "scientist",
        "Student, Medizin": "student, medicine",
        "Student, Naturwissenschaften": "student, natural sciences",
        "Veterinärmediziner": "veterinarian",
        "MTA": "medical technical assistant",
    },
    ("RTKonzepte", "RTKonzept"): {
        "d1 6 Gy - Mamma": "d1 6 Gy - mammary gland",
        "d1 6 Gy - Femora": "d1 6 Gy - femora",
        "Mamma01 d1 3,0 Gy": "Mammary01 d1 3.0 Gy",
        "Mamma02 d1 10,0 Gy": "Mammary02 d1 10.0 Gy",
        **{
            v: v.replace(",", ".")
            for v in [
                "d1-5 10,0 Gy",
                "d1 10,0 Gy",
                "d1 15,0 Gy",
                "d1-4 10.0 Gy",
                "d1-3 10.0 Gy",
                "d1-2 10.0 Gy",
                "d1 2.0 Gy",
                "d1-10 3,0 Gy",
                "d1-5 2.0 Gy",
                "d1 3.0 Gy",
                "d1-10 2.0 Gy",
                "d1-5 3.0 Gy",
                "d1-5 10,0 Gy 225 kV",
                "d1 10 Gy225 kV",
                "d1-11 12x2,0 Gy",
                "d1-6 6x2,0 Gy",
            ]
        },
    },
    ("RTKonzepte", "PTV"): {"Tumor": "tumour", "Beide Femora": "both femora"},
    ("RTKonzepte", "Bemerkung"): {
        "Sonderfall 10 x 2,0Gy +2x2,0Gy": "Special case 10 x 2.0 Gy + 2 x 2.0 Gy",
        "Vorversuch Olaparib": "Pilot olaparib",
        **_COLLAB,
        **_same(
            *[
                f"Standard {x}"
                for x in [
                    "5 x 10 Gy",
                    "1 x 10 Gy",
                    "4 x 10 Gy",
                    "3 x 10 Gy",
                    "2 x 10 Gy",
                    "10 x 3 Gy",
                    "5 x 2 Gy",
                    "1 x 3 Gy",
                    "10 x 2 Gy",
                    "5 x 3 Gy",
                    "1 x 10 Gy 225 kV",
                    "6 x 2 Gy",
                ]
            ]
        ),
    },
    ("Rollen", "RolleName"): {
        "K": "C-start",
        "E": "C-end",
        "V": "Pilot",
        "B": "Obs",
        "KT": "NoTumour",
        "OP": "Surgery",
        "Abbruch": "Abort",
        "NNZ": "Unassigned",
        "OE": "OrganRem",
        "MP": "Macro",
        "NP": "Nano",
        **_same("RT", "RChT", "ChT"),
    },
    ("Rollen", "Bemerkung"): {
        "Radiatio": "radiotherapy",
        "Radio-Chemotherapie": "radiochemotherapy",
        "Chemotherapie": "chemotherapy",
        "Kontrolltier am Anfang": "control animal, killed at the start",
        "Kontrolltier am Ende": "control animal, killed at the end",
        "Vorversuch": "pilot experiment",
        "Beobachtung ohne Therapie": "observation without therapy",
        "Kein TU lokal trotz Injektion oder vorzeitiger Abbruch": "no local tumour despite injection, or early termination",
        "Operative Entfernung Primärtumor": "surgical removal of the primary tumour",
        "Versuchsabbruch: AZ / Spontaner Tod": "termination: general condition / spontaneous death",
        "Tier noch nicht zugeordnet": "animal not yet assigned",
        "Organentnahme": "organ removal",
        "Injektion von Makrophagen": "injection of macrophages",
        "Injektion von Nanopartikeln": "injection of nanoparticles",
    },
    ("SeitenBez", "Seite"): {"unbekannt": "unknown", "links": "left", "rechts": "right", "beidseits": "both sides"},
    ("Standorte", "NameStandort"): {
        "Versuchstierhaltung Campus Forschung": "Animal facility, research campus",
        "Strahlenbunker": "Radiation bunker",
    },
    ("Standorte", "Raum"): {"2. Tür links": "2nd door on the left", "101": "101"},
    ("Standorte", "Rack"): {"Links neben dem Abzug": "left of the fume hood", "IVC-Rack": "IVC rack"},
    ("Tagescheck", "Einfahrmodus"): {"Monat": "month", "Tag": "day", "Woche": "week"},
    ("Tagescheck", "Fehler"): {"keine": "none"},
    ("Tiertypen", "Tierart"): {"Maus": "mouse", "Mensch": "human"},
    ("Tiertypen", "Tiertyp"): {
        "Humanoid": "human",
        **_same(
            "PFP/RAG2",
            "SCID",
            "SCID Select",
            "Balb/c",
            "FVB/N-Tg(MMTV-PyVT)",
            "NSG NOD.Cg-Prkdcscid",
            "C57Bl/6",
            "NMRI-Foxn1nu",
            "C57Bl/6J",
        ),
    },
    ("Tiertypen", "Sicherheitsstufe"): _SAFETY,
    ("Tumorzelllinien", "Sicherheitsstufe"): _SAFETY,
    ("Tiertypen", "Bemerkung"): (
        _breeders := {
            "UKE-Zucht": "UKE in-house breeding",
            "Quelle der Tumorzellen": "source of the tumour cells",
            "Wildtyp Loges": "wild type (Loges)",
            "Brustkrebsmodell": "breast cancer model",
            "UKE-Zucht-Kooperation": "UKE breeding, collaboration",
            **_same("Charles River", "Servier", "Janvier"),
        }
    ),
    ("Zuechter", "Bemerkung"): _breeders,
    ("Tierversuche", "Projekttitel"): _PROJECT_TITLES,
    ("Tierversuche", "Bemerkung"): {
        "SF fragen für Details": "ask SF for details",
        "Kooperation Neurochirurgie": "Collaboration neurosurgery",
        "Kooperation DESY": "Collaboration DESY",
        "Kooperationsprojekt Lebertumoren": "Collaboration project liver tumours",
        "Organentnahmeantrag für ALU-PCR / Genehmigung Dr. Tiemann am 23.04.2025": "Organ-removal application for ALU-PCR / approved by Dr. Tiemann on 2025-04-23",
        **_COLLAB,
        **_same("TAS102", "UCCH"),
    },
    ("Todesursachen", "Todesursache"): {
        "Unbekannt": "Unknown",
        "Versuch regulär beendet": "Experiment ended as planned",
        "Tötung bei Tumorulcus": "Killed: tumour ulceration",
        "Tötung bei schlechtem AZ": "Killed: poor general condition",
        "Spontaner Tod": "Spontaneous death",
        "Unbekannt - von Pflege entsorgt": "Unknown: disposed of by animal care",
        "Schlechter AZ / V.a. Lymphom": "Poor general condition / suspected lymphoma",
        "Tod direkt nach OP": "Death directly after surgery",
    },
    ("Toetungsarten", "Toetungsart"): {
        "C02/O2": "CO2/O2",
        "C02": "CO2",
        "Cervicale Dislokation": "Cervical dislocation",
        "Spontaner Tod": "Spontaneous death (not killed)",
        "Finale Blutentnahme": "Final blood sampling",
        "Finale Ketaminnarkose": "Final ketamine anaesthesia",
    },
    ("Topographie", "Bemerkung"): {
        "Injektion der Tumorzellen subkutan in der rechten Flanke.": "s.c. tumour cell injection, right flank",
        "Injektion der Tumorzellen subkutan in der rechten Seite (Thoraxende).": "s.c. tumour cell injection, right side (end of thorax)",
        "Primärtumor": "primary tumour",
        "Lunge bei Sektionen": "lung at necropsy",
        "Leber bei Sektionen": "liver at necropsy",
        "Knochenmark rechter Femur bei Sektion": "bone marrow right femur at necropsy",
        "Knochenmark rechter Tibia bei Sektion": "bone marrow right tibia at necropsy",
        "Knochenmark Femur und Tibia bei Sektion (rechtes Bein)": "bone marrow femur + tibia at necropsy (right leg)",
        "Großhirn bei Sektion": "cerebrum at necropsy",
        "Darmpaket bei der Sektion.": "intestine at necropsy",
        "Muskel": "muscle",
        "Nieren bei der Sektion": "kidneys at necropsy",
        "Injektion der Tumorzellen subcutan subscapulär": "s.c. tumour cell injection, subscapular",
        "Stereotaktische Hirninjektion": "stereotactic brain injection",
        "Milz bei Sektion": "spleen at necropsy",
    },
    ("Traeger", "NameTraeger"): {
        "NaCl 0,9%": "NaCl 0.9%",
        "Ethanol 51% in Stammlösung": "Ethanol 51% in stock solution",
        "PBS": "PBS",
        "HPMC 0,5%": "HPMC 0.5%",
    },
    ("Traeger", "Bemerkung"): {"Wird weiter verdünnt mit NaCl 0,9%": "further diluted with NaCl 0.9%"},
    ("Tumorzelllinien", "NameZelllinie"): (
        _cells := {
            "H69AR Luziferase": "H69AR luciferase",
            "PC-3 RGB + Luziferase": _LUC,
            "PC-3DR RGB + Luziferase": _LUC_DR,
            "Plasmidinduktion CTNNB1 P53 MYC-LUC": "Plasmid induction CTNNB1 P53 MYC-LUC",
            **_same(
                "OH-1 RGB",
                "H69 RGB",
                "LNCaP RGB",
                "PC-3 RGB",
                "VCaP RGB",
                "PC-3",
                "HT29",
                "PymT",
                "MCF7",
                "OH1",
                "H69AR RGB",
                "C5",
                "HCT116",
                "CACO-2",
                "SW48",
                "GL261",
                "NCH644OBC",
            ),
        }
    ),
    ("Tumorzelllinien", "Bemerkung"): {
        "Vorversuch Injektion Feldhaus": "Pilot injection Feldhaus",
        "OH1 in Matrigel": "OH1 in Matrigel",
        "Früher OH1, RGB in Matrigel": "formerly OH1, RGB in Matrigel",
        "Früher OH1, Luziferase in Matrigel": "formerly OH1, luciferase in Matrigel",
        "TAS-Projekt": "TAS project",
        "RGB + Luziferase": "RGB + luciferase",
        "Glioblastom": "glioblastoma",
        "PC-3 Docetaxel-resistent": "PC-3 docetaxel-resistant",
        "Durch Plasmid induzierte Tumoren": "plasmid-induced tumours",
        **_COLLAB,
        **_same("RGB in Matrigel", "RGB 8x", "RGB 4x", "RGB 32x"),
    },
    ("VGruppe", "Zellen"): {
        **_cells,
        "keine": "none",
        "H69AR RGB Luziferase": "H69AR RGB luciferase",
        "GBM": "GBM",
        "Ruthenium-Glutamin": "Ruthenium glutamine",
        "Vorversuch N68/2024 Erste Gruppe": "Pilot N68/2024 first group",
        "PCR-RGB": "PCR-RGB",
        "Makrophagen": "macrophages",
        "Nanopartikel": "nanoparticles",
        "OH-1": "OH-1",
        **{f"PC-3 RGB + Luziferase{i}": f"{_LUC} {i}" for i in range(1, 5)},
        **{f"PC-3DR RGB + Luziferase{i}": f"{_LUC_DR} {i}" for i in range(1, 5)},
    },
    ("Versuchsreihen", "NameReihe"): {
        "Vorversuch 1": "Pilot 1",
        "Test der Zelllmigration unter Bestrahlung": "Cell migration under irradiation",
        "Hauptversuch 1": "Main experiment 1",
        "Vorversuch Loges 1": "Pilot Loges 1",
        "Brustkrebsmodell": "Breast cancer model",
        "Nanopartikel": "Nanoparticles",
        "Vorversuch 1 Zellmigration": "Pilot 1 cell migration",
        "Vorversuch 1 Olaparib": "Pilot 1 olaparib",
        "Hauptversuch TAS102": "Main experiment TAS102",
        "Vorversuch N127/2019 PC3": "Pilot N127/2019 PC3",
        "Hauptversuch N127/2019 PC3": "Main experiment N127/2019 PC3",
        "Vorversuch PC3": "Pilot PC3",
        "Hauptversuch PC3": "Main experiment PC3",
        "Vorversuch N006/2020": "Pilot N006/2020",
        "Hauptversuch N006/2020": "Main experiment N006/2020",
        "Hauptversuch N052/2020": "Main experiment N052/2020",
        "Hauptversuch N029/2023": "Main experiment N029/2023",
        "Vorversuch N068/2024": "Pilot N068/2024",
        "Organentnahme für ALU-PCR": "Organ removal for ALU-PCR",
    },
    ("Versuchsreihen", "Versuchsziel1"): {
        "Dosisfindung": "dose finding",
        "Analyse der Migration von in-vivo gelabelten Zellen in ein bestrahltes Areal": "migration of in-vivo labelled cells into an irradiated area",
        "Der Einfluss von Bestrahlung, Chemotherapie, simultaner Radiochemotherapie und Operation auf die "
        "Metastasierung in Xenograftmausmodellen": "influence of RT, ChT, simultaneous RChT and surgery on metastasis in xenograft mouse models",
        "Resistenz und Knochenmarkmikrometastasierung bei antiandrogener Therapie des Mammakarzinoms": "resistance and bone-marrow micrometastasis under anti-androgen therapy of breast cancer",
        "Methoden lernen": "learning methods",
        "Untersuchung der Tumorspezifischen Strahlensensibilisierung mittels gerichteter Inhibierung der DNA-Reparatur": "tumour-specific radiosensitisation by targeted inhibition of DNA repair",
        "MRT-Bildgebung Tumoren": "MRI of tumours",
        "Analyse der Migration von Tumorzellen": "migration of tumour cells",
        "Wirkung von Olaparib": "effect of olaparib",
        "Wirksamkeit von TAS102": "efficacy of TAS102",
        "Der Einfluss von Bestrahlung, Chemotherapie, Radio-Chemotherapie und Operation auf die Metastasierung": "influence of RT, ChT, RChT and surgery on metastasis",
        "Immunmodulation beim GBM": "immunomodulation in glioblastoma",
        "Ansprechen Metastasen": "response of metastases",
        "Zellverfolgung": "cell tracking",
        "Test des Lebertumormodells": "test of the liver tumour model",
        "Gewebegewinnung Referenzproben ALU-PCR": "tissue for ALU-PCR reference samples",
    },
    ("Versuchsreihen", "Versuchsziel2"): {
        "Verträglichkeit": "tolerability",
        "Klonale Selektionierung der Metastasen": "clonal selection of metastases",
        "Handling der Tiere": "animal handling",
        "Änderungen durch Nanopartikel": "changes due to nanoparticles",
        "Ansprechen auf die Radiatio im Luziferase-Signal": "radiotherapy response in the luciferase signal",
        "Test von Zellen und Geweben": "test of cells and tissues",
        "Verträglichkeit TAS102": "tolerability of TAS102",
        "Ansprechen Hirntumoren": "response of brain tumours",
        "Abscopale Strahleneffekte": "abscopal radiation effects",
    },
    ("Versuchsreihen", "Versuchsziel3"): {
        "Tumorwachstum": "tumour growth",
        "Änderungen durch lokale Radiatio": "changes due to local radiotherapy",
        "Strahlenreaktion umliegender Normalgewebe": "radiation response of surrounding normal tissue",
        "Methodentest": "method test",
        "Immunreaktionen": "immune reactions",
    },
    ("Versuchstiere", "Geschlecht"): {"m": "m", "w": "f"},
    ("Blut", "Type"): {"agmaire": "Maire work group"},
    ("Knochenmark", "Type"): {"agmaire": "Maire work group"},
}

# Free text, one entry per record: translated with FREE_TEXT_EN (``--english``), not with a fixed dictionary
FREE_TEXT = {
    ("Bestrahlungen", "Bemerkung"): "remarks on single RT fractions",
    ("Blut", "Bemerkung"): "remarks on blood samples",
    ("Chemotherapie", "Bemerkung"): "remarks on single ChT administrations",
    ("Gewebe", "Bemerkung"): "remarks on tissue samples",
    ("Knochenmark", "Bemerkung"): "remarks on bone-marrow samples",
    ("Kryo", "Bemerkung"): "remarks on cryo samples",
    ("Logbuch", "Bemerkung"): "raw request parameters of the web application (credentials masked)",
    ("MRTSequenzen", "Bemerkung"): "remarks on MRI sequences",
    ("Narkosen", "Bemerkung"): "remarks on anaesthesia sessions",
    ("Operationen", "OpBericht"): "surgery reports",
    ("PCR", "Bemerkung"): "remarks on PCR results",
    ("Paraffin", "Bemerkung"): "remarks on paraffin samples",
    ("Prozeduren", "Befund"): "clinical findings",
    ("Prozeduren", "Prozedur"): "clinical procedures performed",
    ("SystemVariablen", "value"): "configuration values of the web application",
    ("Tagescheck", "Bemerkung"): "remarks on device checks",
    ("Tumor", "Bemerkung"): "remarks on tumour injections",
    ("VGruppe", "Bemerkung"): "description of each injection batch",
    ("Versuchsreihen", "Bemerkung"): "remark on one trail",
    ("Versuchstiere", "Bemerkung"): "remarks on single mice",
    ("Visiten", "Belastung"): "burden notes on single visits",
    ("Visiten", "Bemerkung"): "remarks on single visits",
}

# Language-neutral text: numbers and dates stored as text, codes, ids, URLs, names
NEUTRAL = {
    ("Applikation", "Bemerkung"),
    ("Bestrahlungen", "QualitaetIst"),
    ("Bestrahlungen", "Datum"),
    ("Bestrahlungen", "Beginn"),
    ("Bestrahlungen", "Ende"),
    ("Bestrahlungen", "Unterschrift"),
    ("Blut", "LfdNr"),
    ("Knochenmark", "LfdNr"),
    ("ChTKonzepte", "ChTTage1"),
    ("ChTKonzepte", "ChTTage2"),
    ("ChTKonzepte", "ChTTage3"),
    ("ChemoMed", "IdKlinikFirma"),
    ("ChemoMed", "IdPerson"),
    ("ChemoMed", "Bemerkung"),
    ("Chemotherapie", "ChTDatum"),
    ("Chemotherapie", "ChTBeginn"),
    ("Chemotherapie", "ChTEnde"),
    ("Chemotherapie", "Unterschrift"),
    ("Destinationen", "Destination"),
    ("Destinationen", "Link"),
    ("Druckmessungen", "Einheit"),
    ("Faerbungen", "KurzNameFaerbung"),
    ("Farben", "Bemerkung"),
    ("KlinikFirma", "Hausnummer"),
    ("KlinikFirma", "PLZ"),
    ("KlinikFirma", "Ort"),
    ("Kryo", "Turm"),
    ("Kryo", "Schublade"),
    ("Kryo", "Reihe"),
    ("Kryo", "Spalte"),
    ("Kryo", "LfdNr"),
    ("Logbuch", "Ip"),
    ("Logbuch", "Aktion"),
    ("MRTKontrast", "IdKlinikFirma"),
    ("MRTKontrast", "Bemerkung"),
    ("MRTSequenzen", "MRTScanner"),
    ("MRTWichtungen", "Bemerkung"),
    ("Markierungen", "Bemerkung"),
    ("Messkammern", "Bemerkung"),
    ("Module", "Modullink"),
    ("Module", "Shortcut"),
    ("Module", "Icon"),
    ("Module", "IconGr"),
    ("Module", "Color"),
    ("Narkosen", "NarkoseDatum"),
    ("Narkosen", "NarkoseBeginn"),
    ("Narkosen", "NarkoseEnde"),
    ("OTBoxTypen", "Spalten"),
    ("OTBoxTypen", "SliceProSpalte"),
    ("OTBoxTypen", "OTproSlice"),
    ("OTBoxen", "OTBoxTyp"),
    ("OTBoxen", "Zellen"),
    ("OTBoxen", "LfdNr"),
    ("OTBoxen", "Versuchsreihe"),
    ("OTSchnittKonzept", "Prozeduren"),
    ("OTSchnittProzeduren", "AnzahlSchnitte"),
    ("OTSchnittProzeduren", "Schnitte"),
    ("OTSchnittProzeduren", "Faerbung"),
    ("Objekttraeger", "AnzMet"),
    ("Objekttraeger", "AnzRGB"),
    ("Objekttraeger", "Bemerkung"),
    ("Objekttraeger", "IndexOTOrt"),
    ("Operationen", "DatumOP"),
    ("Operationen", "BeginnOp"),
    ("Operationen", "EndeOp"),
    ("Operationen", "Tumorgewicht"),
    ("Organsysteme", "Bemerkung"),
    ("PCR", "IdGewKmBlut"),
    ("PCR", "LfdNr"),
    ("PCR", "LfdNrTier"),
    ("PCR", "Jahr"),
    ("Paraffin", "Tonne"),
    ("Paraffin", "Schublade"),
    ("Prozeduren", "Datum"),
    ("Prozeduren", "Uhrzeit"),
    ("Qualifikationen", "Bemerkung"),
    ("RTKonzepte", "Qualitaet"),
    ("Standorte", "Bemerkung"),
    ("SystemVariablen", "name"),
    ("Tagescheck", "Datum"),
    ("Tagescheck", "Zeit"),
    ("Tierversuche", "Versuchsnummer"),
    ("Tierversuche", "Ansprechpartner4"),
    ("Tierversuche", "Ansprechpartner5"),
    ("Todesursachen", "Bemerkung"),
    ("Toetungsarten", "Bemerkung"),
    ("Tumor", "TransDatum"),
    ("Tumor", "EntnahmeDatum"),
    ("VGruppe", "Datum"),
    ("Versuchsreihen", "VersuchStart"),
    ("Versuchsreihen", "VersuchEnde"),
    ("Versuchstiere", "IdTBase"),
    ("Versuchstiere", "IdKurz"),
    ("Versuchstiere", "Vater"),
    ("Versuchstiere", "Mutter"),
    ("Versuchstiere", "Geburtsdatum"),
    ("Versuchstiere", "Sterbedatum"),
    ("Versuchstiere", "Zuechter"),
    ("Versuchstiere", "Kaefignummer"),
    ("Visiten", "KGVerlauf"),
    ("Visiten", "Visitendatum"),
    ("Visiten", "Visitenzeit"),
    *[("Visiten", c) for c in _VISIT_ITEMS],
    *[("PCR_Import", c) for c in COLUMNS_EN["PCR_Import"] if c != "Datum"],
}
# Neutral columns whose names may contain German letters (none at the moment)
NEUTRAL_PROPER_NOUNS: set = set()

# Codes "<German prefix><running number>": the prefix is translated, the number kept
CODE_PREFIXES_EN = {
    ("Paraffin", "Histokasten"): {
        "hk_agar_paraffin": "cassette_agar_paraffin",
        "hirn_paraffin": "brain_paraffin",
        "leber_paraffin": "liver_paraffin",
        "tumor_paraffin": "tumour_paraffin",
        "leber_rgb": "liver_rgb",
        "tumor_rgb": "tumour_rgb",
    },
}

# Names of the request parameters in the application log (Logbuch.Bemerkung); "selrep_<n>" is kept
LOG_PARAMS_EN = {
    **{
        k: k
        for k in [
            "Button1",
            "Service",
            "Sessionid",
            "Submit1",
            "active",
            "air",
            "airpress",
            "arm",
            "blood",
            "blood2",
            "bonemarrow",
            "bonemarrow2",
            "box",
            "brain",
            "cells1",
            "cells2",
            "cells3",
            "check_new",
            "chemo",
            "chemo_new",
            "cp1",
            "cp2",
            "cp3",
            "date",
            "day",
            "dose",
            "export",
            "facs",
            "fieldx",
            "fieldy",
            "file_id",
            "follow",
            "hr",
            "hr1",
            "hr2",
            "hre",
            "hrn",
            "hrnn",
            "id",
            "id1",
            "id2",
            "id3",
            "id4",
            "id5",
            "index",
            "laser",
            "line",
            "liver",
            "login",
            "min",
            "min1",
            "min2",
            "mine",
            "minn",
            "minnn",
            "mon",
            "n2",
            "name",
            "nday",
            "new",
            "new_report",
            "nh",
            "nhr1",
            "nhr2",
            "nmin1",
            "nmin2",
            "nmon",
            "nyr",
            "o2",
            "organ",
            "organ_selected",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "password",
            "proc_new",
            "project",
            "protocol",
            "pv1",
            "pv2",
            "pv3",
            "pv4",
            "pv5",
            "radio",
            "radio_new",
            "ref",
            "regulatory",
            "rgb",
            "row",
            "scode",
            "scores",
            "selected",
            "service",
            "sessID",
            "sessionid",
            "setservice",
            "sid",
            "spleen",
            "start",
            "temp",
            "trizol",
            "user",
            "username",
            "volume",
            "yr",
            "yrh",
            "MU",
            "isolation",
        ]
    },
    "1l": "marking1_left",
    "1r": "marking1_right",
    "2l": "marking2_left",
    "2r": "marking2_right",
    "n1l": "n_marking1_left",
    "n1r": "n_marking1_right",
    "n2l": "n_marking2_left",
    "n2r": "n_marking2_right",
    "aktid": "active_mouse_id",
    "analg": "analgesic",
    "anblasen": "beh_reaction_to_blowing",
    "anfrage": "request",
    "ani_new": "animal_new",
    "ani_upd": "animal_update",
    "apathie": "beh_apathy",
    "appl": "route",
    "applikationen": "routes",
    "art": "kind",
    "atmung": "cond_breathing",
    "augen": "cond_eyes",
    "ausgabe": "output",
    "automutilation": "beh_self_mutilation",
    "basedataani": "animal_base_data",
    "befund": "finding",
    "bel": "burden_note",
    "bem": "remark",
    "bemalu": "remark_alu",
    "bemblood": "remark_blood",
    "bembonemarrow": "remark_bone_marrow",
    "bemlibrain": "remark_left_brain",
    "bemlilu": "remark_left_lung",
    "bemliver": "remark_liver",
    "bemn": "remark_new",
    "bempara": "remark_paraffin",
    "bemrebrain": "remark_right_brain",
    "bemrelu": "remark_right_lung",
    "bemrgb": "remark_rgb",
    "bemtumor": "remark_tumour",
    "benutzer": "users",
    "beruehrung": "beh_reaction_to_touch",
    "boxen": "boxes",
    "dateconcept": "date_protocol",
    "dist": "distance",
    "dosis": "dose",
    "dosis1": "dose1",
    "dosis2": "dose2",
    "eday": "removal_day",
    "edit_act_concept": "edit_protocol",
    "einfahr": "mode",
    "einschub": "slot",
    "emon": "removal_month",
    "eyr": "removal_year",
    "fehler": "error",
    "fell": "cond_fur",
    "gesamtliste": "full_list",
    "gew": "weight",
    "gewalu": "weight_alu",
    "gewebeart": "tissue_type",
    "gewicht": "weight",
    "gewpara": "weight_paraffin",
    "gewrgb": "weight_rgb",
    "gewtumor": "weight_tumour",
    "gruppe": "group",
    "haltung": "cond_posture",
    "hatcher": "breeder",
    "histofarbe": "histology_colour",
    "histologie": "histology",
    "histonr": "histology_number",
    "hydratation": "cond_hydration",
    "idarm": "arm_id",
    "iday": "injection_day",
    "idconcept": "protocol_id",
    "idgrp": "group_id",
    "idgruppe": "group_id",
    "idseite": "side_id",
    "idtherapy": "therapy_id",
    "idtier": "mouse_id",
    "idtierdir": "mouse_id_direct",
    "idtiersel": "mouse_id_selected",
    "idtopo": "site_id",
    "idvisite": "visit_id",
    "idzelllinie": "cell_line_id",
    "imon": "injection_month",
    "ins_new_concept": "insert_new_protocol",
    "iyr": "injection_year",
    "kammer": "chamber",
    "keinenark": "no_anaesthesia",
    "koerperoeffnungen": "cond_body_orifices",
    "koordination": "beh_coordination",
    "kraempfe": "cond_cramps",
    "kryo": "cryo",
    "kryogel": "cryo_gel",
    "kryojue": "cryo_juecker",
    "kurz": "short_id",
    "kurzid": "short_id",
    "lade": "drawer",
    "laehmungen": "cond_paralysis",
    "lfdnr": "running_number",
    "lfdnrtier": "running_number_mouse",
    "libraing": "left_brain_g",
    "lichtfeld": "light_field",
    "lilu": "left_lung",
    "lilug": "left_lung_g",
    "linie": "line",
    "linien": "lines",
    "liverg": "liver_g",
    "mann": "male",
    "markierungen": "markings",
    "med": "drug",
    "motorik": "beh_motor_function",
    "muskeltonus": "cond_muscle_tone",
    "mutter": "mother",
    "narkot": "anaesthetic",
    "neugier": "beh_curiosity",
    "operateur": "surgeon",
    "opok": "surgery_ok",
    "para": "paraffin",
    "personen": "people",
    "post": "next",
    "preve": "previous",
    "prozedur": "procedure",
    "ptvist": "target_volume",
    "qual": "beam",
    "rebraing": "right_brain_g",
    "reihe": "trail",
    "relu": "right_lung",
    "relug": "right_lung_g",
    "schlaf": "beh_sleep",
    "schmerz": "beh_pain",
    "sect": "section",
    "selarm": "selected_arm",
    "selbased": "selected_base_data",
    "selcl": "selected_cell_line",
    "selimpl": "selected_implantation",
    "selscore": "selected_score",
    "sozialkontakte": "beh_social_contact",
    "standort": "location",
    "tbase": "tbase_id",
    "temperatur": "cond_temperature",
    "therapie": "therapy",
    "tierid": "mouse_id",
    "tieridcurr": "current_mouse_id",
    "tiernr": "mouse_number",
    "tiersel": "mouse_selected",
    "tiertyp": "strain",
    "tiertypen": "strains",
    "tierversuche": "projects",
    "todesursachen": "death_causes",
    "toetungsarten": "kill_methods",
    "tonne": "container",
    "tul_del": "tumour_implant_delete",
    "tul_new": "tumour_implant_new",
    "tul_upd": "tumour_implant_update",
    "tumor": "tumour",
    "tumorimpl": "tumour_implantation",
    "turm": "tower",
    "tuvolumen": "tumour_volume",
    "tuvolumen2": "tumour2_volume",
    "upd": "update",
    "ursache": "cause",
    "vater": "father",
    "verhalten": "beh_behaviour",
    "versuch": "experiment",
    "vis_new": "visit_new",
    "vis_new_stapel": "visit_new_batch",
    "vis_next": "visit_next",
    "vis_prev": "visit_previous",
    "vis_upd": "visit_update",
    "visitealerts": "visit_alerts",
    "visiteidcurr": "current_visit_id",
    "visiten": "visits",
    "visitenedit": "visit_edit",
    "visiteneu": "visit_add",
    "visitestapel": "visit_batch",
    "zuechter": "breeders",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a MySQL dump of the mouse database to a pickle.")
    parser.add_argument("--dump", default=NEW_DUMP)
    parser.add_argument("--out", default=NEW_PICKLE)
    parser.add_argument("--check-only", action="store_true", help="Only run the self-check on the old dump.")
    parser.add_argument("--force", action="store_true", help="Overwrite --out if it exists.")
    parser.add_argument(
        "--english", action="store_true", help=f"Write the English copy of {NEW_PICKLE} to {NEW_PICKLE_EN} instead."
    )
    args = parser.parse_args()

    self_check()
    if args.check_only:
        return
    if args.english:
        if Path(NEW_PICKLE_EN).exists() and not args.force:
            raise SystemExit(f"{NEW_PICKLE_EN} exists, use --force to overwrite it")
        old = pd.read_pickle(OLD_PICKLE)
        for name, real in OLD_LONGTEXT.items():
            old[name] = old[name].rename(columns={"longtext": real})
            old[name][real] = old[name][real].str.replace("\\r", "\r").str.replace("\\n", "\n")
        free_text = load_free_text()
        translate(old, free_text)  # the old export must be translatable too (old vs new comparison)
        german = pd.read_pickle(NEW_PICKLE)
        tables = translate(german, free_text)
        pd.to_pickle(tables, NEW_PICKLE_EN)
        print(f"Wrote {NEW_PICKLE_EN}: {len(tables)} tables. Free-text columns (translated with {FREE_TEXT_EN}):")
        print(free_text_columns(tables, german).to_string(index=False))
        return
    if Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} exists, use --force to overwrite it")
    old = pd.read_pickle(OLD_PICKLE)
    tables = read_dump(args.dump, like=old)
    print("Columns not cast to the old pickle's dtype (would change values):")
    print(dtype_differences(tables, old).to_string(index=False))
    pd.to_pickle(tables, args.out)
    print(f"Wrote {args.out}: {len(tables)} tables, " + ", ".join(f"{n} {len(t)}" for n, t in tables.items()))


if __name__ == "__main__":
    main()
