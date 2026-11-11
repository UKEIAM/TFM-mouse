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
converts ``mausprojekt-3.sql`` to ``mausprojekt-3.pandas.pkl``.

Known quirks of the old pickle, tolerated by the self-check: the ``longtext``
columns ``Operationen.OpBericht`` and ``Prozeduren.Befund`` are named
``longtext`` there and keep ``\\r\\n`` escaped. Here they get their real names
and are unescaped.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

OLD_DUMP = "/data/mice/maus_mysql.sql"
OLD_PICKLE = "/data/mice/maus.pandas.pkl"
NEW_DUMP = "/data/mice/mausprojekt-3.sql"
NEW_PICKLE = "/data/mice/mausprojekt-3.pandas.pkl"

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
            raise ValueError(f"Cannot parse the dump at {pos}: {text[pos:pos + 80]!r}")
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
        for name, frame in tables.items() if name in reference
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a MySQL dump of the mouse database to a pickle.")
    parser.add_argument("--dump", default=NEW_DUMP)
    parser.add_argument("--out", default=NEW_PICKLE)
    parser.add_argument("--check-only", action="store_true", help="Only run the self-check on the old dump.")
    parser.add_argument("--force", action="store_true", help="Overwrite --out if it exists.")
    args = parser.parse_args()

    self_check()
    if args.check_only:
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
