import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
import io
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file using PyPDF2 and OCR (pytesseract) if needed, per page.
    Accepts a file-like object or bytes.
    Returns the extracted text as a string.
    Raises Exception if extraction fails.
    """
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        pdf_bytes = file.read() if not hasattr(
            file, "read") or not callable(file.read) else file.read()
        if hasattr(file, "read") and callable(file.read):
            # If file is an async UploadFile, await read
            try:
                pdf_bytes = file.read()
            except TypeError:
                import asyncio
                pdf_bytes = asyncio.run(file.read())
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        # Convert all pages to images for possible OCR
        images = convert_from_bytes(pdf_bytes)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip() and len(text.strip()) > 20:
                extracted_pages.append(text.strip())
            else:
                # Fallback to OCR for this page with preprocessing
                if i < len(images):
                    img = images[i]
                    # Convert to grayscale
                    img = img.convert("L")
                    # Increase contrast
                    img = ImageEnhance.Contrast(img).enhance(2.0)
                    # Apply adaptive thresholding (binarization)
                    img = img.point(lambda x: 0 if x < 180 else 255, '1')
                    # Optionally denoise (unsharp mask)
                    # Ensure mode is 'L' for pytesseract
                    img = img.convert("L")
                    img = img.filter(ImageFilter.UnsharpMask(
                        radius=2, percent=150, threshold=3))
                    ocr_text = pytesseract.image_to_string(img)
                    extracted_pages.append(ocr_text.strip())
                else:
                    extracted_pages.append("")
        full_text = "\n".join(extracted_pages)
        if not full_text.strip():
            raise Exception("No text extracted from PDF (even with OCR).")
        return full_text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    pdf_dir = Path("documents")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in 'documents' directory.")
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            class DummyFile:
                def __init__(self, content):
                    self._content = content

                def read(self):
                    return self._content

                def seek(self, pos):
                    pass
            dummy_file = DummyFile(f.read())
            try:
                text = extract_text_from_pdf(dummy_file)
                print(
                    f"[TextParser] {pdf_file.name}: {len(text)} chars extracted")
                print(f"  Preview: {text[:200]}...\n")
            except Exception as e:
                print(f"[TextParser] ERROR extracting {pdf_file.name}: {e}")
