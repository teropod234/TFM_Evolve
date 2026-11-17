from os import path
from src.utils.data_processing import transform_pdf_to_csv

def process_data(pdf_path, csv_path):
    """
    Process the data by cleaning and transforming the PDF to CSV.
    """
    transform_pdf_to_csv(pdf_path, csv_path)

if __name__ == "__main__":
    process_data("data/raw/data.pdf", "data/processed/data.csv")