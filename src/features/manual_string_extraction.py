#!/usr/bin/env python3
"""
Manual feature extraction from C function code strings.

This module provides functions to extract various string-based features from C code
that might be relevant for vulnerability detection. These features focus on code structure,
programming patterns, and potential security-relevant aspects of the code.
"""

import re
from typing import Dict, List, Set, Tuple, Any
import math

import pandas as pd
from scipy.sparse import csr_matrix


def extract_basic_code_metrics(code_str: str) -> Dict[str, float]:
    """
    Extracts basic code metrics such as:
    - Line count
    - Character count
    - Characters per line
    - Blank line ratio
    - Comment ratio
    - Documentation density

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of extracted metrics with feature names as keys and their values
    """
    # Get all lines and strip trailing whitespace
    lines = code_str.split("\n")
    clean_lines = [line.strip() for line in lines]

    # Basic metrics
    char_count = len(code_str)
    line_count = len(clean_lines)

    # Blank line detection
    blank_lines = sum(1 for line in clean_lines if not line)
    blank_line_ratio = blank_lines / max(line_count, 1)

    # Comment detection
    comment_lines = sum(1 for line in clean_lines if line.startswith("//") or line.startswith("/*"))
    comment_line_ratio = comment_lines / max(line_count, 1)

    # Characters per line (excluding blank lines)
    non_blank_lines = max(line_count - blank_lines, 1)
    chars_per_line = char_count / non_blank_lines

    # Documentation detection (javadoc/doxygen style comments)
    doc_pattern = r"/\*\*|\*\s@\w+|/\*\*\*|\*\s[A-Z][A-Za-z]*:|@[a-z]+\s"
    doc_matches = len(re.findall(doc_pattern, code_str))
    doc_density = doc_matches / max(char_count / 100, 1)  # Normalize per 100 chars

    return {
        "line_count": float(line_count),
        "char_count": float(char_count),
        "chars_per_line": float(chars_per_line),
        "blank_line_ratio": float(blank_line_ratio),
        "comment_ratio": float(comment_line_ratio),
        "doc_density": float(doc_density),
    }


def extract_control_flow_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to control flow complexity:
    - Cyclomatic complexity (McCabe complexity)
    - Condition count (if/else/switch/case)
    - Loop count and density
    - Nesting depth
    - Return statement count
    - Goto statement count

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of control flow metrics with feature names as keys and their values
    """
    # Control flow keywords
    if_count = len(re.findall(r"\bif\s*\(", code_str))
    else_if_count = len(re.findall(r"\belse\s+if\s*\(", code_str))
    else_count = len(re.findall(r"\belse\s*\{", code_str))
    case_count = len(re.findall(r"\bcase\s+[^:]+:", code_str))
    default_count = len(re.findall(r"\bdefault\s*:", code_str))

    # Loops
    for_count = len(re.findall(r"\bfor\s*\(", code_str))
    while_count = len(re.findall(r"\bwhile\s*\(", code_str))
    do_while_count = len(re.findall(r"\bdo\s*\{", code_str))

    # Adjust if count (exclude else-if)
    if_count -= else_if_count

    # Other control flow
    switch_count = len(re.findall(r"\bswitch\s*\(", code_str))
    goto_count = len(re.findall(r"\bgoto\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;", code_str))
    return_count = len(re.findall(r"\breturn\b", code_str))

    # Logical operators (used in conditions)
    logical_and_count = len(re.findall(r"&&", code_str))
    logical_or_count = len(re.findall(r"\|\|", code_str))

    # Calculate cyclomatic complexity:
    # 1 + (conditions + loops + logical operators)
    cyclomatic_complexity = (
        1
        + if_count
        + else_if_count
        + case_count
        + for_count
        + while_count
        + do_while_count
        + logical_and_count
        + logical_or_count
    )

    # Total conditions and loops
    condition_count = if_count + else_if_count + else_count + case_count + default_count
    loop_count = for_count + while_count + do_while_count

    # Estimate nesting depth by calculating max consecutive {
    lines = code_str.split("\n")
    max_depth = 0
    current_depth = 0
    for line in lines:
        current_depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, current_depth)

    # Get approximate code length for density calculations
    code_length = len(code_str)
    # Avoid division by zero
    code_length_100 = max(code_length / 100, 1)

    return {
        "cyclomatic_complexity": float(cyclomatic_complexity),
        "condition_count": float(condition_count),
        "loop_count": float(loop_count),
        "condition_density": float(condition_count / code_length_100),
        "loop_density": float(loop_count / code_length_100),
        "max_nesting_depth": float(max_depth),
        "return_count": float(return_count),
        "goto_count": float(goto_count),
        "switch_count": float(switch_count),
    }


def extract_function_characteristics(code_str: str) -> Dict[str, float]:
    """
    Extracts characteristics related to function declaration and structure:
    - Return type indicators (void, pointer, primitive types)
    - Parameter count
    - Function visibility/storage modifiers
    - Function length

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of function characteristics with feature names as keys and their values
    """
    # Known C key terms
    key_terms = {
        "void",
        "int",
        "char",
        "float",
        "double",
        "short",
        "long",
        "signed",
        "unsigned",
        "struct",
        "union",
        "enum",
        "static",
        "extern",
        "register",
        "volatile",
        "const",
        "auto",
        "inline",
    }

    # Check if function has a valid signature
    if "(" not in code_str:
        return {"has_valid_signature": 0.0, "param_count": 0.0, "is_recursive": 0.0}

    # Extract function signature
    signature = code_str.split("(")[0].strip()

    # Get function name (last word in signature)
    parts = signature.split()
    func_name = parts[-1].strip("*") if parts else ""

    # Check if function is recursive (calls itself)
    # Look for function name followed by opening parenthesis
    is_recursive = 1.0 if re.search(rf"\b{re.escape(func_name)}\s*\(", code_str[len(signature) :]) else 0.0

    # Get parameter section
    param_section = ""
    if "(" in code_str and ")" in code_str:
        param_section = code_str.split("(", 1)[1].split(")", 1)[0]

    # Count parameters by counting commas + 1, unless it's void or empty
    param_count = 0.0
    if param_section and not param_section.strip() == "void":
        # Handle multi-line parameters
        param_section = re.sub(r"\s+", " ", param_section)
        param_count = float(param_section.count(",") + 1)

    # Function length (approximate by brace matching)
    first_brace = code_str.find("{")
    if first_brace >= 0:
        function_length = len(code_str) - first_brace
    else:
        function_length = len(code_str)

    return {
        "has_valid_signature": 1.0,
        "param_count": param_count,
        "is_recursive": is_recursive,
        "function_length": float(function_length),
    }


def extract_security_risk_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to potential security risks:
    - Unsafe string/memory functions
    - Memory allocation/deallocation
    - Pointer arithmetic
    - Variable declarations without initialization
    - Format string usage
    - Error handling patterns
    - Use of assert

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of security-related metrics with feature names as keys and their values
    """
    # Define lists of potentially dangerous functions
    unsafe_str_funcs = ["strcpy", "strcat", "sprintf", "gets", "scanf", "vsprintf", "strncpy", "strncat"]

    memory_funcs = ["malloc", "calloc", "realloc", "free", "alloca"]

    # Count occurrences of unsafe string functions
    unsafe_str_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in unsafe_str_funcs)

    # Count memory allocation/deallocation
    mem_alloc_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in memory_funcs[:4])
    free_count = len(re.findall(r"\bfree\s*\(", code_str))

    # Detect potential memory leaks (allocations > free)
    potential_mem_leak = float(max(mem_alloc_count - free_count, 0))

    # Detect pointer arithmetic
    ptr_arithmetic = len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-]\s*[\d]+", code_str))

    # Detect uninitialized variables (type name followed by variable without =)
    c_types = [
        "int",
        "char",
        "float",
        "double",
        "void",
        "unsigned",
        "long",
        "short",
        "struct",
        "enum",
        "union",
        "bool",
        "size_t",
        "ssize_t",
    ]
    uninit_vars = 0
    for c_type in c_types:
        # Find variable declarations without initialization
        uninit_vars += len(re.findall(rf"\b{c_type}\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]*\])?\s*;", code_str))

    # Format string vulnerabilities
    format_funcs = ["printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "vsprintf"]
    format_strings = sum(
        len(re.findall(rf"\b{func}\s*\(\s*[^,)]*,\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,)]", code_str)) for func in format_funcs
    )

    # Error handling patterns
    error_handling = len(re.findall(r"(error|err|errno|perror|strerror|exit|abort|assert)", code_str, re.IGNORECASE))

    # Use of assertions
    assert_count = len(re.findall(r"\bassert\s*\(", code_str))

    # Array access without bounds checking
    array_access = len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\[[^\]]*\]", code_str))

    return {
        "unsafe_str_func_count": float(unsafe_str_count),
        "mem_alloc_count": float(mem_alloc_count),
        "free_count": float(free_count),
        "potential_mem_leak": potential_mem_leak,
        "pointer_arithmetic": float(ptr_arithmetic),
        "uninitialized_vars": float(uninit_vars),
        "format_string_usage": float(format_strings),
        "error_handling_count": float(error_handling),
        "assert_count": float(assert_count),
        "array_access_count": float(array_access),
    }


def extract_variable_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to variables and data types:
    - Count of primitive types (int, char, etc.)
    - Pointer usage
    - Array usage
    - Local variable count
    - Struct/Union usage
    - Type casting frequency
    - Use of sizeof

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of variable-related features with names as keys and values
    """
    # Common C primitive types
    primitive_types = [
        "int",
        "char",
        "float",
        "double",
        "void",
        "unsigned",
        "long",
        "short",
        "bool",
        "size_t",
        "ssize_t",
    ]

    # Count variable declarations by primitive type
    type_counts = {}
    for c_type in primitive_types:
        type_counts[f"{c_type}_count"] = float(len(re.findall(rf"\b{c_type}\b", code_str)))

    # Pointer usage
    pointer_count = len(re.findall(r"\w+\s*\*\s*\w+", code_str)) + code_str.count("*")

    # Array declarations
    array_decl = len(re.findall(r"\w+\s+\w+\s*\[[^\]]*\]", code_str))

    # Count local variable declarations (approximation)
    # This might catch function parameters as well
    local_var_pattern = (
        r"(?:int|char|float|double|unsigned|long|short|bool|size_t|void|struct|enum|union)\s+[a-zA-Z_][a-zA-Z0-9_]*"
    )
    local_vars = len(re.findall(local_var_pattern, code_str))

    # Struct/Union usage
    struct_count = len(re.findall(r"\bstruct\b", code_str))
    union_count = len(re.findall(r"\bunion\b", code_str))

    # Type casting
    type_cast = len(
        re.findall(r"\(\s*(?:int|char|float|double|unsigned|long|short|void|bool|size_t)\s*\*?\s*\)", code_str)
    )

    # Use of sizeof
    sizeof_count = len(re.findall(r"\bsizeof\s*\(", code_str))

    # Combine all features
    features = {
        "pointer_count": float(pointer_count),
        "array_decl_count": float(array_decl),
        "local_var_count": float(local_vars),
        "struct_count": float(struct_count),
        "union_count": float(union_count),
        "type_cast_count": float(type_cast),
        "sizeof_count": float(sizeof_count),
    }

    # Add all individual type counts
    features.update(type_counts)

    return features


def extract_api_usage_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to API usage that might indicate vulnerability patterns:
    - System call usage
    - File operations
    - Network functions
    - String manipulation
    - Memory operations
    - Random number generation
    - Cryptography functions

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of API usage features with names as keys and values
    """
    # System call related functions
    syscall_funcs = [
        "system",
        "exec",
        "popen",
        "fork",
        "execve",
        "execl",
        "execlp",
        "execle",
        "execv",
        "execvp",
        "execvpe",
        "_popen",
        "_wpopen",
        "CreateProcess",
    ]

    # File operations
    file_funcs = [
        "fopen",
        "open",
        "read",
        "write",
        "fread",
        "fwrite",
        "fprintf",
        "fscanf",
        "fclose",
        "fseek",
        "ftell",
        "rewind",
        "fflush",
        "fileno",
    ]

    # Network functions
    net_funcs = [
        "socket",
        "connect",
        "bind",
        "listen",
        "accept",
        "send",
        "recv",
        "sendto",
        "recvfrom",
        "gethostbyname",
        "getaddrinfo",
    ]

    # String manipulation
    str_funcs = [
        "strcpy",
        "strncpy",
        "strcat",
        "strncat",
        "strcmp",
        "strncmp",
        "strlen",
        "strchr",
        "strrchr",
        "strstr",
        "strtok",
    ]

    # Memory operations
    mem_funcs = ["memcpy", "memmove", "memset", "memcmp", "malloc", "calloc", "realloc", "free", "alloca"]

    # Random number generation
    rand_funcs = ["rand", "random", "srand", "srandom"]

    # Cryptography functions
    crypto_funcs = ["md5", "sha1", "sha256", "aes", "des", "blowfish", "crypt"]

    # Count occurrences of each API category
    syscall_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in syscall_funcs)
    file_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in file_funcs)
    net_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in net_funcs)
    str_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in str_funcs)
    mem_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in mem_funcs)
    rand_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in rand_funcs)
    crypto_count = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in crypto_funcs)

    return {
        "syscall_count": float(syscall_count),
        "file_operation_count": float(file_count),
        "network_count": float(net_count),
        "string_function_count": float(str_count),
        "memory_function_count": float(mem_count),
        "random_function_count": float(rand_count),
        "crypto_function_count": float(crypto_count),
    }


def extract_error_handling_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to error handling patterns:
    - Error checking patterns
    - Return value validation
    - Use of errno
    - NULL pointer checks
    - Exception-like patterns

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of error handling features with names as keys and values
    """
    # Error return value checking
    err_check_patterns = [
        r"if\s*\(\s*.+\s*==\s*(-1|NULL|0)\s*\)",
        r"if\s*\(\s*!\s*.+\s*\)",
        r"if\s*\(\s*.+\s*!=\s*(-1|NULL|0)\s*\)",
    ]

    error_checks = sum(len(re.findall(pattern, code_str)) for pattern in err_check_patterns)

    # Errno usage
    errno_usage = len(re.findall(r"\berrno\b", code_str))
    perror_usage = len(re.findall(r"\bperror\s*\(", code_str))
    strerror_usage = len(re.findall(r"\bstrerror\s*\(", code_str))

    # NULL pointer checks
    null_checks = (
        len(re.findall(r"if\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*==\s*NULL\s*\)", code_str))
        + len(re.findall(r"if\s*\(\s*NULL\s*==\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)", code_str))
        + len(re.findall(r"if\s*\(\s*![a-zA-Z_][a-zA-Z0-9_]*\s*\)", code_str))
    )

    # Error return statements
    error_returns = len(re.findall(r"return\s+(-1|NULL|0)\s*;", code_str))

    # Goto for error handling (goto err or goto error or goto fail, etc.)
    goto_err = len(re.findall(r"goto\s+(err|error|fail|cleanup|out|exit)", code_str, re.IGNORECASE))

    # Try-catch like structures (do-while(0) with breaks for errors)
    try_patterns = len(re.findall(r"do\s*{.*?}\s*while\s*\(\s*0\s*\)", code_str, re.DOTALL))

    return {
        "error_check_count": float(error_checks),
        "errno_usage": float(errno_usage + perror_usage + strerror_usage),
        "null_check_count": float(null_checks),
        "error_return_count": float(error_returns),
        "goto_error_count": float(goto_err),
        "try_catch_pattern_count": float(try_patterns),
    }


def extract_code_style_features(code_str: str) -> Dict[str, float]:
    """
    Extracts features related to code style that might correlate with code quality:
    - Indentation consistency
    - Bracket style
    - Comment style
    - Variable naming style
    - Spacing patterns

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary of code style features with names as keys and values
    """
    lines = code_str.split("\n")
    indentations = []

    # Calculate indentation statistics
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            indentations.append(indent)

    # Indentation consistency (standard deviation)
    if len(indentations) > 1:
        avg_indent = sum(indentations) / len(indentations)
        indent_variance = sum((x - avg_indent) ** 2 for x in indentations) / len(indentations)
        indent_consistency = 1.0 / (1.0 + indent_variance)  # Higher value means more consistent
    else:
        indent_consistency = 1.0  # Default for short functions

    # Bracket style (K&R vs. Allman)
    # K&R: if (x) {
    # Allman: if (x)
    #         {
    kr_style = sum(1 for line in lines if re.search(r"[)\s]\s*{", line))
    allman_style = sum(
        1 for i in range(1, len(lines)) if re.search(r"^\s*{", lines[i]) and not re.search(r"{", lines[i - 1])
    )

    # Bracket style ratio (higher means more K&R, lower means more Allman)
    total_brackets = kr_style + allman_style
    bracket_style_ratio = kr_style / max(total_brackets, 1)

    # Comment style (// vs /* */)
    single_line_comments = sum(1 for line in lines if "//" in line)
    multiline_comments = code_str.count("/*")

    # Comment style ratio (higher means more //, lower means more /* */)
    total_comments = single_line_comments + multiline_comments
    comment_style_ratio = single_line_comments / max(total_comments, 1)

    # Variable naming style
    snake_case = len(re.findall(r"\b[a-z][a-z0-9]*(_[a-z0-9]+)+\b", code_str))
    camel_case = len(re.findall(r"\b[a-z][a-z0-9]*([A-Z][a-z0-9]*)+\b", code_str))

    # Naming style ratio (higher means more snake_case, lower means more camelCase)
    total_naming = snake_case + camel_case
    naming_style_ratio = snake_case / max(total_naming, 1)

    # Spacing patterns
    spaces_after_control = sum(1 for line in lines if re.search(r"\b(if|for|while|switch)\s+\(", line))
    no_spaces_after_control = sum(1 for line in lines if re.search(r"\b(if|for|while|switch)\(", line))

    # Spacing style ratio (higher means more spaces after control statements)
    total_controls = spaces_after_control + no_spaces_after_control
    spacing_style_ratio = spaces_after_control / max(total_controls, 1)

    return {
        "indent_consistency": float(indent_consistency),
        "bracket_style_ratio": float(bracket_style_ratio),
        "comment_style_ratio": float(comment_style_ratio),
        "naming_style_ratio": float(naming_style_ratio),
        "spacing_style_ratio": float(spacing_style_ratio),
    }


def analyze_token_patterns(code_str: str) -> Dict[str, float]:
    """
    Analyzes token patterns that might be indicative of poor coding practices
    or potential vulnerabilities including:
    - Multiple statements on one line (e.g., a; b;)
    - Empty blocks (e.g., if (x) {})
    - Assignment in condition (e.g., if (x = 5))
    - Magic numbers (numeric literals in code)

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary containing token pattern analysis features
    """
    # Multiple statements on same line (look for ; followed by non-whitespace, non-comment)
    multi_statements = len(re.findall(r";[^;\n/]*;", code_str))

    # Empty blocks
    empty_blocks = len(re.findall(r"{[\s]*}", code_str))

    # Assignment in condition
    assign_in_condition = len(re.findall(r"if\s*\([^=]*=[^=][^)]*\)", code_str))

    # Magic numbers (numeric literals not in initialization, exclude 0, 1, -1 which are often legitimate)
    magic_numbers = len(re.findall(r'[^a-zA-Z0-9_"\'](?:[2-9]|[1-9][0-9]+)(?![0-9])', code_str))

    # Detect common input validation patterns
    input_validation = len(re.findall(r"if\s*\([^)]*(?:len|size|count|length)[^)]*(?:>|<|>=|<=|==)[^)]*\)", code_str))

    return {
        "multi_statement_lines": float(multi_statements),
        "empty_block_count": float(empty_blocks),
        "assignment_in_condition": float(assign_in_condition),
        "magic_number_count": float(magic_numbers),
        "input_validation_count": float(input_validation),
    }


def extract_buffer_use_patterns(code_str: str) -> Dict[str, float]:
    """
    Extracts patterns related to buffer usage that could indicate potential
    buffer overflow vulnerabilities:
    - Uses of fixed-size buffers
    - String operations without explicit bounds checks
    - Copy operations with potential overflow risks

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary containing buffer usage pattern features
    """
    # Fixed-size buffer declarations
    fixed_buffers = len(
        re.findall(r"(?:char|unsigned char|wchar_t|TCHAR)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*\d+\s*\]", code_str)
    )

    # Potentially dangerous string operations (without size checks)
    unsafe_str_ops = len(re.findall(r"\b(?:strcpy|strcat|sprintf|gets)\s*\(", code_str))

    # Safe string operations (with explicit size parameters)
    safe_str_ops = len(re.findall(r"\b(?:strncpy|strncat|snprintf|strncmp)\s*\(", code_str))

    # Buffer size check patterns
    size_check_patterns = len(re.findall(r"if\s*\([^)]*(?:size|len|sizeof)[^)]*(?:<|<=|>|>=)[^)]*\)", code_str))

    # Use of dangerous functions that write to buffers
    buffer_write_funcs = ["sprintf", "gets", "strcpy", "strcat", "scanf", "fscanf", "vsprintf", "memcpy"]
    dangerous_writes = sum(len(re.findall(rf"\b{func}\s*\(", code_str)) for func in buffer_write_funcs)

    return {
        "fixed_buffer_count": float(fixed_buffers),
        "unsafe_str_ops": float(unsafe_str_ops),
        "safe_str_ops": float(safe_str_ops),
        "buffer_size_checks": float(size_check_patterns),
        "dangerous_buffer_writes": float(dangerous_writes),
    }


def extract_complexity_metrics(code_str: str) -> Dict[str, float]:
    """
    Calculates more advanced complexity metrics beyond basic cyclomatic complexity:
    - Halstead complexity measures (effort, difficulty)
    - Program length and vocabulary measures
    - Fan-in/fan-out approximation (function call metrics)

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary containing advanced complexity metric features
    """
    # Get unique operators and operands for Halstead metrics
    # This is a simplified approximation
    operators = re.findall(r"[+\-*/%&|^~!=<>?:;,\.\[\]\(\)\{\}]|->|==|!=|<=|>=|&&|\|\||\+\+|\-\-", code_str)
    operands = re.findall(r"\b(?:(?:[a-zA-Z_][a-zA-Z0-9_]*)|(?:[0-9]+(?:\.[0-9]+)?))\b", code_str)

    # Count unique operators and operands
    n1 = len(set(operators))  # Unique operators
    n2 = len(set(operands))  # Unique operands
    N1 = len(operators)  # Total operators
    N2 = len(operands)  # Total operands

    # Calculate Halstead metrics
    try:
        program_length = N1 + N2
        vocabulary = n1 + n2
        volume = program_length * (math.log2(vocabulary) if vocabulary > 0 else 0)
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
    except:
        # Default values if calculation fails
        program_length = float(len(code_str))
        vocabulary = float(len(set(code_str)))
        volume = 0.0
        difficulty = 0.0
        effort = 0.0

    # Fan-out: count of function calls made by this function
    # Simple approximation by counting patterns like name(...)
    function_calls = len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\([^;{]*\)", code_str))

    return {
        "halstead_length": float(program_length),
        "halstead_vocabulary": float(vocabulary),
        "halstead_volume": float(volume),
        "halstead_difficulty": float(difficulty),
        "halstead_effort": float(effort),
        "function_calls_count": float(function_calls),
    }


def extract_function_type(code_str: str) -> Dict[str, float]:
    """
    Extracts the function type (return type and modifiers) and creates binary features
    for common return types and modifiers.

    Args:
        code_str: C function code as a string

    Returns:
        Dictionary containing function type features as binary indicators
    """
    # Set of known C key terms
    key_terms = {
        "void",
        "int",
        "char",
        "float",
        "double",
        "short",
        "long",
        "signed",
        "unsigned",
        "struct",
        "union",
        "enum",
        "static",
        "extern",
        "register",
        "volatile",
        "const",
        "auto",
        "inline",
        "bool",
        "size_t",
        "ssize_t",
    }

    features = {}

    # Default values for all type features
    for term in key_terms:
        features[f"is_{term}"] = 0.0

    # Check if function has a valid signature
    if "(" not in code_str:
        return features

    # Extract everything before the first parenthesis
    before_paren = code_str.split("(", 1)[0].strip()

    # Split into words
    type_parts = before_paren.split()
    if len(type_parts) < 2:
        return features  # Not enough parts for valid function signature

    # Remove function name (last word before parenthesis)
    func_type = " ".join(type_parts[:-1])

    # Check for presence of each key term in the function type
    for term in key_terms:
        if re.search(rf"\b{term}\b", func_type):
            features[f"is_{term}"] = 1.0

    return features


def manual_string_extraction(dataframe: pd.DataFrame) -> Tuple[csr_matrix, List[str]]:
    """
    Extracts all features from a given C function code string and returns a sparse matrix and feature labels.

    Args:
        dataframe: DataFrame containing the C function code strings in a 'func' column

    Returns:
        Tuple[csr_matrix, List[str]]: Sparse matrix of extracted features (rows: functions, columns: features), and list of feature labels
    """
    feature_dicts = []
    for code_str in dataframe["x_string"]:
        features = {}
        features.update(extract_basic_code_metrics(code_str))
        features.update(extract_control_flow_features(code_str))
        features.update(extract_function_characteristics(code_str))
        features.update(extract_security_risk_features(code_str))
        features.update(extract_variable_features(code_str))
        features.update(extract_api_usage_features(code_str))
        features.update(extract_error_handling_features(code_str))
        features.update(extract_code_style_features(code_str))
        features.update(analyze_token_patterns(code_str))
        features.update(extract_buffer_use_patterns(code_str))
        features.update(extract_complexity_metrics(code_str))
        features.update(extract_function_type(code_str))
        feature_dicts.append(features)

    # Convert to DataFrame for consistent column order
    features_df = pd.DataFrame(feature_dicts).fillna(0.0)
    feature_labels = list(features_df.columns)
    # Convert to sparse matrix
    feature_matrix = csr_matrix(features_df.values)
    return feature_matrix, feature_labels
