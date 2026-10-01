import os
import sys
import glob

# Add the project root to the system path so Python can find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transform.sec_cleaner import clean_sec_filing


def process_filing(input_path: str, output_path: str) -> None:
    # 1. Extract
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_markdown = f.read()

    # 2. Transform
    clean_markdown = clean_sec_filing(raw_markdown)

    # 3. Load
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_markdown)


if __name__ == "__main__":
    # Input directory: parsed markdown files produced by the upstream parser.
    # The script is expected to run from the project root, so this relative
    # path resolves to <project_root>/parsed_docs/markdown/
    markdown_dir = os.path.join("parsed_docs", "markdown")

    # Output directory: cleaned files land here.
    # Keeping it separate from the input dir makes re-runs safe — the glob
    # below will never accidentally pick up previously cleaned files even if
    # the suffix check were removed.
    output_dir = os.path.join("parsed_docs", "markdown_clean")
    os.makedirs(output_dir, exist_ok=True)

    # Find all raw .md files (skip any _clean.md that may have slipped in)
    md_files = [
        p for p in glob.glob(os.path.join(markdown_dir, "*.md"))
        if not p.endswith("_clean.md")
    ]

    if not md_files:
        print(f"No markdown files found in '{markdown_dir}'.")
    else:
        for input_file in md_files:
            basename      = os.path.basename(input_file)
            clean_name    = basename.replace(".md", "_clean.md")
            output_file   = os.path.join(output_dir, clean_name)

            print(f"Processing : {basename}")
            print(f"  Input    : {input_file}")
            print(f"  Output   : {output_file}")

            process_filing(input_file, output_file)
            print(f"  Done.\n")

        print("All files processed successfully!")