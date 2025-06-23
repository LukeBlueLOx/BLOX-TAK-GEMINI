#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Libraries / Biblioteki ---

# For filesystem operations, e.g., creating paths and checking existence.
# Do operacji na systemie plików, np. tworzenia ścieżek i sprawdzania istnienia.
import os

# For using regular expressions to split text based on patterns.
# Do używania wyrażeń regularnych do dzielenia tekstu na podstawie wzorców.
import re

# For creating and managing logs (to console and file).
# Do tworzenia i zarządzania logami (do konsoli i pliku).
import logging
from logging.handlers import RotatingFileHandler

# LangChain's standard data structure for a piece of text with metadata.
# Standardowa struktura danych LangChain dla fragmentu tekstu z metadanymi.
from langchain.schema.document import Document

# LangChain's tool for splitting large texts into smaller, overlapping chunks.
# Narzędzie LangChain do dzielenia dużych tekstów na mniejsze, nakładające się fragmenty.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# A robust library for extracting text and data from PDF files.
# Solidna biblioteka do wyciągania tekstu i danych z plików PDF.
import pdfplumber


def setup_logger(log_folder: str):
    """
    Sets up a logger to output to both console and a rotating file.
    Konfiguruje logger do zapisu zarówno do konsoli, jak i do rotacyjnego pliku.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Prevents adding multiple handlers if the function is called more than once
    # Zapobiega dodawaniu wielu handlerów, jeśli funkcja zostanie wywołana więcej niż raz
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    # Handler konsoli
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    # Handler pliku
    fh = RotatingFileHandler(
        os.path.join(log_folder, 'rag_preprocessor.log'),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3
    )
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# Initialize logger here, assuming a LOGS folder exists at the project root for now
# Inicjalizacja loggera, zakładając na razie folder LOGS w głównym katalogu projektu
logger = setup_logger('LOGS')


def process_legal_document(file_path: str) -> list[Document]:
    """
    Processes a structured legal PDF, splitting it by articles.
    Przetwarza ustrukturyzowany prawny plik PDF, dzieląc go na artykuły.
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processing legal document: {file_name}")
    print(f"English: Processing legal document: {file_name}\nPolski: Przetwarzam dokument prawny: {file_name}")

    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        logger.error(f"Could not read PDF file {file_path}: {e}")
        print(
            f"English: ERROR - Could not read PDF file {file_path}: {e}\nPolski: BŁĄD - Nie można odczytać pliku PDF {file_path}: {e}")
        return []

    chunks = re.split(r'(?=\nArt\.\s*\d+[a-zA-Z]?\.?)', full_text)

    documents = []
    for chunk in chunks:
        if not chunk.strip():
            continue

        match = re.search(r'^(Art\.\s*\d+[a-zA-Z]?\.?)', chunk)
        article_num = match.group(1).strip() if match else "N/A"

        doc = Document(
            page_content=chunk.strip(),
            metadata={
                "source": file_name,
                "article": article_num,
                "type": "legal_corpus"
            }
        )
        documents.append(doc)

    logger.info(f"  > Found and processed {len(documents)} articles/fragments.")
    print(
        f"English:   > Found and processed {len(documents)} articles/fragments.\nPolski:   > Znaleziono i przetworzono {len(documents)} artykułów/fragmentów.")
    return documents


def process_case_document_archive(file_path: str) -> list[Document]:
    """
    Processes a 'Combined_Content' PDF. It splits the content back into its
    original source files in memory, then chunks them.
    Przetwarza plik PDF typu 'Combined_Content'. Dzieli zawartość z powrotem na
    oryginalne pliki w pamięci, a następnie je chunkuje.
    """
    archive_name = os.path.basename(file_path)
    logger.info(f"Processing case document archive: {archive_name}")
    print(
        f"English: Processing case document archive: {archive_name}\nPolski: Przetwarzam archiwum dokumentów sprawy: {archive_name}")

    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        logger.error(f"Could not read PDF archive file {file_path}: {e}")
        print(
            f"English: ERROR - Could not read PDF archive file {file_path}: {e}\nPolski: BŁĄD - Nie można odczytać pliku archiwum PDF {file_path}: {e}")
        return []

    parts = re.split(r'--- BEGINNING OF FILE: (.*?) ---', full_text)

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

    for i in range(1, len(parts), 2):
        original_filename = parts[i].strip()
        content = parts[i + 1].split('--- END OF FILE:')[0].strip()

        logger.info(f"  > Extracting virtual file: {original_filename}")
        print(
            f"English:   > Extracting virtual file: {original_filename}\nPolski:   > Ekstrahuję wirtualny plik: {original_filename}")

        if not content:
            logger.warning(f"    > Content for {original_filename} is empty. Skipping.")
            print(
                f"English:     > Content for {original_filename} is empty. Skipping.\nPolski:     > Zawartość dla {original_filename} jest pusta. Pomijam.")
            continue

        chunks = text_splitter.split_text(content)

        for chunk_content in chunks:
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "source_original": original_filename,
                    "source_archive": archive_name,
                    "type": "case_corpus"
                }
            )
            documents.append(doc)

    logger.info(f"  > Extracted {len(documents)} total chunks from the archive.")
    print(
        f"English:   > Extracted {len(documents)} total chunks from the archive.\nPolski:   > Wyekstrahowano {len(documents)} wszystkich chunków z archiwum.")
    return documents


def get_documents_from_source(source_folder: str, corpus_type: str) -> list[Document]:
    """
    Main orchestrator function for the preprocessing step.
    Główna funkcja orkiestrująca dla kroku preprocessingu.
    """
    all_processed_docs = []

    logger.info(f"Starting data preprocessing for corpus '{corpus_type}' from folder: {source_folder}")
    print(
        f"\nEnglish: Starting data preprocessing for corpus '{corpus_type}' from folder: {source_folder}\nPolski: Rozpoczynam preprocessing danych dla korpusu '{corpus_type}' z folderu: {source_folder}")

    if not os.path.exists(source_folder):
        logger.error(f"Source folder not found: {source_folder}")
        print(
            f"English: ERROR - Source folder not found: {source_folder}\nPolski: BŁĄD - Folder źródłowy nie znaleziony: {source_folder}")
        return []

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(source_folder, file_name)
            if corpus_type == 'legal':
                processed_docs = process_legal_document(file_path)
            elif corpus_type == 'case':
                processed_docs = process_case_document_archive(file_path)
            else:
                logger.warning(f"Unknown corpus type '{corpus_type}'. Skipping file {file_name}.")
                print(
                    f"English: WARNING - Unknown corpus type '{corpus_type}'. Skipping file {file_name}.\nPolski: OSTRZEŻENIE - Nieznany typ korpusu '{corpus_type}'. Pomijam plik {file_name}.")
                processed_docs = []
            all_processed_docs.extend(processed_docs)

    logger.info(f"Finished preprocessing for '{corpus_type}'. Total documents prepared: {len(all_processed_docs)}")
    print(
        f"English: Finished preprocessing for '{corpus_type}'. Total documents prepared: {len(all_processed_docs)}\nPolski: Zakończono preprocessing dla '{corpus_type}'. Łączna liczba przygotowanych dokumentów: {len(all_processed_docs)}\n")
    return all_processed_docs