import re
from os import path
from pathlib import Path

import pandas as pd
import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "src" / "data" / "raw"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "src" / "data" / "clean"

COLUMN_NAMES = [
    "posicion",
    "nombre_plan",
    "valor_unidad_cuenta",
    "rentabilidad_30_anios",
    "rentabilidad_25_anios",
    "rentabilidad_20_anios",
    "rentabilidad_15_anios",
    "rentabilidad_10_anios",
    "rentabilidad_5_anios",
    "rentabilidad_3_anios",
    "rentabilidad_1_anio",
    "participes",
    "beneficiarios",
    "aportaciones",
    "prestaciones",
    "aportacion_neta",
    "cuenta_posicion",
    "variacion_trimestre",
    "variacion_anual",
    "nombre_fondo",
    "grupo_financiero",
]

NUMERIC_COLUMNS = [
    "valor_unidad_cuenta",
    "rentabilidad_30_anios",
    "rentabilidad_25_anios",
    "rentabilidad_20_anios",
    "rentabilidad_15_anios",
    "rentabilidad_10_anios",
    "rentabilidad_5_anios",
    "rentabilidad_3_anios",
    "rentabilidad_1_anio",
    "participes",
    "beneficiarios",
    "aportaciones",
    "prestaciones",
    "aportacion_neta",
    "cuenta_posicion",
    "variacion_trimestre",
    "variacion_anual",
]


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """
    Preprocess the data by handling missing values and normalizing.

    Parameters:
    data (pd.DataFrame): The input data to preprocess.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    # Handle missing values
    data = data.fillna(data.mean())

    # Normalize the data
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

    return data


def _clean_text(value):
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _parse_spanish_number(value):
    text = _clean_text(value)
    if text is None:
        return None

    text = text.replace(" ", "")
    if not re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|[+-]?\d+(?:,\d+)?", text):
        return None

    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") == 1 and len(text.split(".")[1]) == 3:
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return None


def _extract_report_metadata(first_row):
    cells = [_clean_text(cell) for cell in first_row]
    title = next((cell for cell in cells if cell and "PLANES DE PENSIONES" in cell), None)
    categoria = None
    if title and " - " in title:
        categoria = title.split(" - ", 1)[1]
        categoria = re.sub(r"\s*\(importe.*$", "", categoria).strip()

    date_parts = [cell for cell in cells if cell and re.search(r"\d|octubre|noviembre|diciembre|enero", cell, re.I)]
    fecha = " ".join(date_parts).strip() or None
    fecha = re.sub(r"^.*?(\d{1,2}\s+de\s+\w+\s+del\s+\d{4})$", r"\1", fecha) if fecha else None
    if fecha and not re.search(r"\d{4}", fecha):
        fecha = " ".join(date_parts).strip()

    return {
        "categoria": categoria,
        "fecha_informe": fecha,
        "titulo": title,
    }


def _is_data_row(row):
    nombre_plan = _clean_text(row[1] if len(row) > 1 else None)
    valor_unidad = _parse_spanish_number(row[3] if len(row) > 3 else None)
    if not nombre_plan or valor_unidad is None:
        return False
    if "nombre" in nombre_plan.lower() and "planes" in nombre_plan.lower():
        return False
    return True


def _row_to_record(row, metadata, posicion):
    nombre_plan = _clean_text(row[1])
    nombre_fondo = _clean_text(row[21] if len(row) > 21 else None)
    grupo_financiero = _clean_text(row[22] if len(row) > 22 else None)

    numeric_values = {
        "valor_unidad_cuenta": _parse_spanish_number(row[3] if len(row) > 3 else None),
        "rentabilidad_30_anios": _parse_spanish_number(row[4] if len(row) > 4 else None),
        "rentabilidad_25_anios": _parse_spanish_number(row[5] if len(row) > 5 else None),
        "rentabilidad_20_anios": _parse_spanish_number(row[6] if len(row) > 6 else None),
        "rentabilidad_15_anios": _parse_spanish_number(row[7] if len(row) > 7 else None),
        "rentabilidad_10_anios": _parse_spanish_number(row[8] if len(row) > 8 else None),
        "rentabilidad_5_anios": _parse_spanish_number(row[9] if len(row) > 9 else None),
        "rentabilidad_3_anios": _parse_spanish_number(row[10] if len(row) > 10 else None),
        "rentabilidad_1_anio": _parse_spanish_number(row[11] if len(row) > 11 else None),
        "participes": _parse_spanish_number(row[12] if len(row) > 12 else None),
        "beneficiarios": _parse_spanish_number(row[13] if len(row) > 13 else None),
        "aportaciones": _parse_spanish_number(row[14] if len(row) > 14 else None),
        "prestaciones": _parse_spanish_number(row[15] if len(row) > 15 else None),
        "aportacion_neta": _parse_spanish_number(row[16] if len(row) > 16 else None),
        "cuenta_posicion": _parse_spanish_number(row[17] if len(row) > 17 else None),
        "variacion_trimestre": _parse_spanish_number(row[19] if len(row) > 19 else None),
        "variacion_anual": _parse_spanish_number(row[20] if len(row) > 20 else None),
    }

    record = {
        "posicion": posicion,
        "nombre_plan": nombre_plan,
        **numeric_values,
        "nombre_fondo": nombre_fondo,
        "grupo_financiero": grupo_financiero,
        **metadata,
    }
    return record


def pdf_to_dataframe(pdf_path):
    """
    Extract the INVERCO pension-plan table from a PDF into a DataFrame.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    records = []
    metadata = {
        "categoria": None,
        "fecha_informe": None,
        "titulo": None,
        "fuente_pdf": pdf_path.name,
    }
    next_posicion = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            table = tables[0]
            if table and metadata["titulo"] is None:
                metadata.update(_extract_report_metadata(table[0]))
                metadata["fuente_pdf"] = pdf_path.name

            for row in table:
                if not _is_data_row(row):
                    continue

                raw_posicion = _parse_spanish_number(row[0] if row else None)
                posicion = int(raw_posicion) if raw_posicion is not None else next_posicion
                records.append(_row_to_record(row, metadata, posicion))
                next_posicion = posicion + 1

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        return dataframe

    ordered_columns = COLUMN_NAMES + ["categoria", "fecha_informe", "titulo", "fuente_pdf"]
    dataframe = dataframe[ordered_columns]
    dataframe["posicion"] = dataframe["posicion"].astype("Int64")
    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def transform_pdf_to_csv(pdf_path, csv_path=None):
    """
    Transform a PDF file to a CSV file.

    Parameters:
    pdf_path (str): The path to the input PDF file.
    csv_path (str): The path to the output CSV file.
                    If omitted, the CSV is written to src/data/clean.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    if csv_path is None:
        csv_path = DEFAULT_CLEAN_DIR / f"{pdf_path.stem}.csv"
    else:
        csv_path = Path(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pdf_to_dataframe(pdf_path)
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def transform_pdfs_to_csv(raw_dir=None, clean_dir=None):
    """
    Convert every PDF in the raw directory into a CSV in data/clean.

    Parameters:
    raw_dir (str): Directory with source PDFs. Defaults to src/data/raw.
    clean_dir (str): Directory for CSV output. Defaults to src/data/clean.

    Returns:
    list[Path]: Paths of the generated CSV files.
    """
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    clean_dir = Path(clean_dir) if clean_dir else DEFAULT_CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        csv_path = clean_dir / f"{pdf_path.stem}.csv"
        transform_pdf_to_csv(pdf_path, csv_path)
        generated.append(csv_path)
    return generated
