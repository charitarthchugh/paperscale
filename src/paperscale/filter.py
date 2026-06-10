import logging
import re
import subprocess
from collections import Counter

from lingua import Language, LanguageDetectorBuilder
from pypdf import PdfReader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PdfFilter:
    def __init__(
        self,
        languages_to_keep=None,
        apply_form_check=True,
        apply_download_spam_check=True,
        download_spam_threshold=0.004,
    ):
        super().__init__()
        self.language_detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        self.languages_to_keep = languages_to_keep if languages_to_keep is not None else [Language.ENGLISH]
        self.apply_form_check = apply_form_check
        self.apply_download_spam_check = apply_download_spam_check
        self.download_spam_threshold = download_spam_threshold

    def _is_form(self, pdf_reader) -> bool:
        # Check if the PDF is a form
        if pdf_reader.get_form_text_fields():
            return True
        return False  # Not a form

    def _is_download_spam(self, base_text: str) -> bool:
        seo_words = {
            "download",
            "pdf",
            "epub",
            "mobi",
            "free",
            "ebook",
            "file",
            "save",
            "casino",
            "viagra",
            "cialis",
            "ciprofloxacin",
        }

        base_text = base_text.strip().lower()
        clean_text = re.sub(r"\W+", " ", base_text)

        word_counts = Counter(clean_text.split())
        total_words = len(clean_text.split())

        if total_words == 0:
            return False

        seo_score = sum(word_counts[word] for word in seo_words if word in word_counts)

        return (seo_score / total_words) > self.download_spam_threshold

    # Returns True if there is something wrong with this PDF
    def filter_out_pdf(self, local_pdf_path: str) -> bool:
        try:
            # Attempt to read the PDF at the beginning
            pdf_reader = PdfReader(local_pdf_path)

            # Form check
            if self.apply_form_check and self._is_form(pdf_reader):
                logger.info(f"Filtering out {local_pdf_path} because it's a form")
                return True  # Filter out
        except Exception as e:
            logger.warning(f"Error reading PDF {local_pdf_path}: {e}")
            return True  # Filter out the PDF if an exception occurs

        # Read the first five pages of text for language calculation
        pdftotext_result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "5", local_pdf_path, "-"],
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if pdftotext_result.returncode != 0:
            logger.warning(f"pdftotext returned {pdftotext_result.returncode} on {local_pdf_path}")
            return True  # Filter out

        base_text = pdftotext_result.stdout.decode("utf-8")

        alpha_count = sum(c.isalpha() for c in base_text)

        if len(base_text) < 200:
            logger.info(f"Keeping {local_pdf_path} on the safe side because not enough text exists in it to analyze")
            return False  # keep the pdf

        if alpha_count / len(base_text) < 0.50:
            logger.info(f"Keeping {local_pdf_path} on the safe side because it's text does not contain many letters so it might be OCRed badly")
            return False  # keep the pdf

        # Language check
        language = self.language_detector.detect_language_of(base_text)
        if language not in self.languages_to_keep:
            logger.info(f"Filtering out {local_pdf_path} because language was {language}")
            return True  # Filter out

        # Download spam check
        if self.apply_download_spam_check and self._is_download_spam(base_text):
            logger.info(f"Filtering out {local_pdf_path} because of SEO/download spam")
            return True  # Filter out

        return False  # Keep the PDF


