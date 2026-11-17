import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "src" / "data" / "raw"
CLEAN_DIR = ROOT / "src" / "data" / "clean"

# Índices de las celdas en la tabla del PDF de INVERCO
COLS = {
    "nombre_plan": 1,
    "valor_unidad_cuenta": 3,
    "rentabilidad_30_anios": 4,
    "rentabilidad_25_anios": 5,
    "rentabilidad_20_anios": 6,
    "rentabilidad_15_anios": 7,
    "rentabilidad_10_anios": 8,
    "rentabilidad_5_anios": 9,
    "rentabilidad_3_anios": 10,
    "rentabilidad_1_anio": 11,
    "participes": 12,
    "beneficiarios": 13,
    "aportaciones": 14,
    "prestaciones": 15,
    "aportacion_neta": 16,
    "cuenta_posicion": 17,
    "variacion_trimestre": 19,
    "variacion_anual": 20,
    "nombre_fondo": 21,
    "grupo_financiero": 22,
}

TEXT_COLS = {"nombre_plan", "nombre_fondo", "grupo_financiero"}


def load_data(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No existe {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(data):
    nums = data.select_dtypes(include=["float64", "int64"]).columns
    data[nums] = data[nums].fillna(data[nums].mean())
    data[nums] = (data[nums] - data[nums].mean()) / data[nums].std()
    return data


def _texto(valor):
    if valor is None:
        return None
    t = " ".join(str(valor).split())
    return t or None


def _numero(valor):
    """Pasa '1.409,50' o '9,0585' a float. Si no es número, None."""
    t = _texto(valor)
    if t is None:
        return None

    t = t.replace(" ", "")
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    elif t.count(".") == 1 and len(t.split(".")[1]) == 3:
        t = t.replace(".", "")  # miles: 12.751

    try:
        return float(t)
    except ValueError:
        return None


def _celda(fila, i):
    return fila[i] if i < len(fila) else None


def _info_informe(fila, nombre_pdf):
    cabecera = " ".join(_texto(c) for c in fila if _texto(c))
    titulo = None
    for c in fila:
        t = _texto(c)
        if t and "PLANES DE PENSIONES" in t:
            titulo = t
            break

    categoria = None
    if titulo and " - " in titulo:
        categoria = titulo.split(" - ", 1)[1]
        categoria = categoria.split("(importe")[0].strip()

    fecha = None
    m = re.search(r"\d{1,2} de \w+ del \d{4}", cabecera)
    if m:
        fecha = m.group(0)

    return {
        "categoria": categoria,
        "fecha_informe": fecha,
        "titulo": titulo,
        "fuente_pdf": nombre_pdf,
    }


def _es_plan(fila):
    nombre = _texto(_celda(fila, COLS["nombre_plan"]))
    valor = _numero(_celda(fila, COLS["valor_unidad_cuenta"]))
    if not nombre or valor is None:
        return False
    # cabecera repetida en cada página
    if "nombre" in nombre.lower() and "planes" in nombre.lower():
        return False
    return True


def pdf_to_dataframe(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No existe {pdf_path}")

    filas = []
    info = None
    n = 1

    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            tablas = pagina.extract_tables()
            if not tablas:
                continue

            tabla = tablas[0]
            if info is None and tabla:
                info = _info_informe(tabla[0], pdf_path.name)

            for fila in tabla:
                if not _es_plan(fila):
                    continue

                pos = _numero(_celda(fila, 0))
                pos = int(pos) if pos is not None else n

                registro = {"posicion": pos}
                for nombre, i in COLS.items():
                    valor = _celda(fila, i)
                    registro[nombre] = _texto(valor) if nombre in TEXT_COLS else _numero(valor)
                registro.update(info)
                filas.append(registro)
                n = pos + 1

    return pd.DataFrame(filas)


def transform_pdf_to_csv(pdf_path, csv_path=None):
    pdf_path = Path(pdf_path)
    csv_path = Path(csv_path) if csv_path else CLEAN_DIR / f"{pdf_path.stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pdf_to_dataframe(pdf_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def transform_pdfs_to_csv(raw_dir=None, clean_dir=None):
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    clean_dir = Path(clean_dir) if clean_dir else CLEAN_DIR

    salidas = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        salidas.append(transform_pdf_to_csv(pdf, clean_dir / f"{pdf.stem}.csv"))
    return salidas
