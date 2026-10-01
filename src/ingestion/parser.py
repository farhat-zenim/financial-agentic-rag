import gc
import json
import logging
from datetime import datetime
from pathlib import Path

# Importations Docling pour le traitement haute précision
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# ==========================================
# 1. LOGGING ET ARBORESCENCE (ALIGNÉE SUR VOTRE PROJET)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Chemin exact extrait de votre capture d'écran : RAG/data/raw_pdfs/pdfs
SOURCE_DIR = Path("./data/raw_pdfs/pdfs")
OUTPUT_BASE = Path("./parsed_docs")

# Dossiers de sortie pour LlamaIndex et la base vectorielle
OUTPUT_MD_DIR = OUTPUT_BASE / "markdown"         # Fichiers MD avec structure de tableaux
OUTPUT_JSON_DIR = OUTPUT_BASE / "json"           # Métadonnées et coordonnées spatiales
OUTPUT_IMG_DIR = OUTPUT_BASE / "images"          # Images et graphiques extraits
MANIFEST_PATH = OUTPUT_BASE / "pipeline_manifest.json" # Suivi du pipeline RAG

# Création des répertoires de sortie
for directory in [OUTPUT_MD_DIR, OUTPUT_JSON_DIR, OUTPUT_IMG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. INITIALISATION DU PIPELINE DOCLING
# ==========================================
pipeline_options = PdfPipelineOptions()

# OCR pour capturer les textes scannés ou intégrés aux graphiques
pipeline_options.do_ocr = True

# Activation du modèle précis TableFormer (indispensable pour les bilans financiers SEC 10-K)
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

# Extraction des images/graphiques avec une résolution de qualité
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

# Instanciation du convertisseur avec nos règles
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# ==========================================
# 3. EXÉCUTION DU PARSING ET SAUVEGARDE
# ==========================================
def execute_rag_ingestion_parsing():
    """
    Parse les fichiers PDF du dossier raw_pdfs/pdfs séquentiellement,
    sauvegarde les sorties MD/JSON/Images et génère le manifeste RAG.
    """
    pdf_files = sorted(list(SOURCE_DIR.glob("*.pdf")))

    if not pdf_files:
        logging.error(f"Aucun fichier PDF trouvé dans {SOURCE_DIR.resolve()}. Vérifiez le dossier.")
        return

    logging.info(f"Début du parsing de {len(pdf_files)} fichiers PDF...")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(pdf_files),
        "documents": []
    }

    for index, pdf_path in enumerate(pdf_files, start=1):
        logging.info(f"[{index}/{len(pdf_files)}] Parsing : {pdf_path.name}")
        base_name = pdf_path.stem

        md_file = OUTPUT_MD_DIR / f"{base_name}.md"
        json_file = OUTPUT_JSON_DIR / f"{base_name}.json"

        doc_entry = {
            "doc_id": base_name,
            "source_pdf": str(pdf_path.resolve()),
            "status": "FAILED",
            "markdown_path": None,
            "json_path": None,
            "page_count": 0,
            "table_count": 0,
            "image_count": 0,
            "images_saved": [],
            "error_log": None
        }

        try:
            # Étape A : Analyse de la mise en page du document
            conv_res = doc_converter.convert(pdf_path)
            doc = conv_res.document

            # Étape B : Exportation Markdown (source pour MarkdownNodeParser de LlamaIndex)
            markdown_content = doc.export_to_markdown()
            with md_file.open("w", encoding="utf-8") as f:
                f.write(markdown_content)
            doc_entry["markdown_path"] = str(md_file.resolve())

            # Étape C : Exportation JSON (conserve les bounding boxes et conteneurs)
            json_content = doc.export_to_dict()
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2)
            doc_entry["json_path"] = str(json_file.resolve())

            # Étape D : Extraction des images et comptage des éléments
            table_count = 0
            image_count = 0

            for element, _level in doc.iterate_items():
                elem_type = element.__class__.__name__

                if elem_type == "TableItem":
                    table_count += 1

                elif elem_type == "PictureItem":
                    image_count += 1
                    img_name = f"{base_name}_img_{image_count}.png"
                    img_path = OUTPUT_IMG_DIR / img_name

                    # Enregistrement de l'image
                    with img_path.open("wb") as fp:
                        element.get_image(doc).save(fp, "PNG")

                    doc_entry["images_saved"].append(str(img_path.resolve()))

            # Mise à jour des informations du document
            doc_entry["status"] = "SUCCESS"
            doc_entry["page_count"] = len(doc.pages) if hasattr(doc, "pages") else 0
            doc_entry["table_count"] = table_count
            doc_entry["image_count"] = image_count

            logging.info(
                f"    -> Succès : Pages={doc_entry['page_count']} | "
                f"Tableaux={table_count} | Images={image_count}"
            )

        except Exception as err:
            logging.error(f"    -> Erreur sur {pdf_path.name}: {err}")
            doc_entry["error_log"] = str(err)

        finally:
            # Étape E : Mise à jour du manifeste au fur et à mesure
            manifest["documents"].append(doc_entry)
            with MANIFEST_PATH.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Étape F : Nettoyage strict de la mémoire RAM/VRAM[cite: 4]
            if 'conv_res' in locals():
                del conv_res
            if 'doc' in locals():
                del doc
            gc.collect()

    logging.info(f"Parsing terminé ! Fichier manifeste disponible : {MANIFEST_PATH.resolve()}")

# ==========================================
# 4. POINT D'ENTRÉE
# ==========================================
if __name__ == "__main__":
    execute_rag_ingestion_parsing()