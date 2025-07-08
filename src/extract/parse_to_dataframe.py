"""
Parse C functions from CSV, deduplicate, preprocess, parse to AST, and save as DataFrame with progress and checkpointing.
"""

import os
import pickle
import subprocess
import tempfile
from typing import Tuple, Optional, Any, List
import pandas as pd

try:
    import pycparser_fake_libc
    import pycparser
    from pycparser import c_ast, c_parser

    FAKE_LIBC_PATH = pycparser_fake_libc.directory
except ImportError as e:
    missing_pkg = "pycparser" if "pycparser" in str(e) else "pycparser_fake_libc"
    print(f"Error: {missing_pkg} package not found. Please install it with:")
    print(f"pip install {missing_pkg}")
    raise

CSV_PATH = "src/data/dataset.csv"
CHECKPOINT_DIR = "src/extract"
CHECKPOINT_PREFIX = "dataframe_checkpoint_"
FUNC_COL = "func"
TARGET_COL = "target"


def preprocess_c_function(func_code: str) -> Tuple[str, bool, str]:
    """
    Preprocess C function code using GCC and fake libc includes.
    Returns (preprocessed_code, success, error_message).
    """
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", encoding="utf-8", delete=False) as temp_file:
        full_code = f"""
#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n{func_code}\n"""
        temp_file.write(full_code)
        temp_file_path = temp_file.name
    try:
        gcc_cmd = ["gcc", "-E", "-std=c99", f"-I{FAKE_LIBC_PATH}", temp_file_path]
        result = subprocess.run(gcc_cmd, capture_output=True, text=True, check=True)
        preprocessed = result.stdout
        os.unlink(temp_file_path)
        return preprocessed, True, ""
    except subprocess.CalledProcessError as e:
        os.unlink(temp_file_path)
        return "", False, f"GCC stderr: {e.stderr}"
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return "", False, f"Unexpected error: {str(e)}"


def clean_preprocessed_code(preprocessed: str) -> str:
    lines = preprocessed.splitlines()
    cleaned_lines = []
    in_code_section = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if not in_code_section and not (stripped.startswith("//") or stripped.startswith("/*")):
            in_code_section = True
        if in_code_section:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def parse_c_code(code: str) -> Tuple[Optional[c_ast.FileAST], bool, str]:
    cleaned_code = clean_preprocessed_code(code)
    if not cleaned_code.strip():
        return None, False, "No code content after cleaning"
    parser = c_parser.CParser()
    try:
        ast = parser.parse(cleaned_code)
        return ast, True, ""
    except Exception as e:
        try:
            wrapped_code = f"int dummy_var;\n{cleaned_code}"
            ast = parser.parse(wrapped_code)
            return ast, True, ""
        except Exception as e2:
            return None, False, f"First: {str(e)}, Second: {str(e2)}"


def save_dataframe_checkpoint(df: pd.DataFrame, checkpoint_num: int) -> None:
    path = f"{CHECKPOINT_DIR}/{CHECKPOINT_PREFIX}{checkpoint_num}.pickle"
    with open(path, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved checkpoint {checkpoint_num} to {path}")


def main() -> None:
    print(f"Reading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, usecols=[FUNC_COL, TARGET_COL])
    print(f"Original rows: {len(df)}")
    df = df.sort_values(by=TARGET_COL, ascending=False)
    df = df.drop_duplicates(subset=[FUNC_COL]).reset_index(drop=True)
    print(f"Rows after deduplication: {len(df)}")
    x_strings: List[str] = []
    x_asts: List[Any] = []
    targets: List[int] = []
    processed_rows = 0
    invalid_rows = 0
    successful_rows = 0
    checkpoint_num = 0
    total = len(df)
    for i, row in df.iterrows():
        try:
            func_code = row[FUNC_COL]
            target = row[TARGET_COL]
        except KeyError as e:
            invalid_rows += 1
            continue
        processed_rows += 1
        if not func_code or not str(func_code).strip():
            invalid_rows += 1
            continue
        try:
            preprocessed, success, error_msg = preprocess_c_function(func_code)
            if not success:
                invalid_rows += 1
                continue
            ast, ast_success, ast_error = parse_c_code(preprocessed)
            if not ast_success:
                invalid_rows += 1
                continue
            x_strings.append(func_code)
            x_asts.append(ast)
            targets.append(int(target))
            successful_rows += 1
        except Exception as e:
            invalid_rows += 1
            continue
        if processed_rows % 100 == 0:
            print(
                f"Progress: {processed_rows}/{total} | Successful: {successful_rows} | Invalid/skipped: {invalid_rows}"
            )
        if processed_rows > 0 and processed_rows % 1000 == 0:
            checkpoint_num += 1
            df_checkpoint = pd.DataFrame({"x_string": x_strings, "x_ast": x_asts, "target": targets})
            save_dataframe_checkpoint(df_checkpoint, checkpoint_num)
    # Final save
    df_final = pd.DataFrame({"x_string": x_strings, "x_ast": x_asts, "target": targets})
    save_dataframe_checkpoint(df_final, checkpoint_num + 1)
    print(f"Done. Total processed: {processed_rows}, valid: {len(df_final)}, skipped: {invalid_rows}")


if __name__ == "__main__":
    main()
