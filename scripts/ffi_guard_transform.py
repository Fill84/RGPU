#!/usr/bin/env python3
"""
Transform Rust FFI functions to use catch_panic wrappers.

Transforms:
    #[no_mangle]
    pub unsafe extern "C" fn func_name(args) -> RetType {
        // body
    }

Into:
    unsafe fn func_name_impl(args) -> RetType {
        // body
    }

    #[no_mangle]
    pub unsafe extern "C" fn func_name(args) -> RetType {
        rgpu_common::ffi::catch_panic(ERROR_VALUE, || func_name_impl(args_names_only))
    }

Usage:
    python ffi_guard_transform.py <input_file> <error_value> [--exclude=func1,func2]
"""

import re
import sys


def find_matching_brace(text, start):
    """Find the matching closing brace for the opening brace at `start`."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def parse_args(args_str):
    """Parse function arguments, returning (full_args, arg_names_only)."""
    if not args_str.strip():
        return "", ""

    # Split by commas but respect nested angle brackets and parens
    args = []
    depth = 0
    current = ""
    for ch in args_str:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        args.append(current.strip())

    # Extract just the names
    names = []
    for arg in args:
        # "name: Type" pattern
        if ':' in arg:
            name = arg.split(':')[0].strip()
            names.append(name)

    return args_str.strip(), ", ".join(names)


def transform_file(input_path, error_value, excludes=None):
    """Transform all #[no_mangle] pub unsafe extern "C" fn in the file."""
    if excludes is None:
        excludes = set()

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern for #[no_mangle] followed by pub unsafe extern "C" fn
    # We need to handle optional attributes between #[no_mangle] and the fn
    pattern = re.compile(
        r'#\[no_mangle\]\s*\n'
        r'(\s*)'  # capture indentation
        r'pub\s+unsafe\s+extern\s+"C"\s+fn\s+'
        r'(\w+)'  # function name
        r'\s*\(([^)]*)\)'  # arguments (simple case, no nested parens in types)
        r'\s*->\s*'
        r'([^\{]+?)'  # return type
        r'\s*\{'
    )

    # Handle more complex argument lists (with nested types)
    # Re-do with a more robust approach
    result = []
    pos = 0

    while pos < len(content):
        # Look for #[no_mangle]
        idx = content.find('#[no_mangle]', pos)
        if idx == -1:
            result.append(content[pos:])
            break

        # Append everything before this match
        result.append(content[pos:idx])

        # Find the function signature
        after_attr = content[idx + len('#[no_mangle]'):]
        # Skip whitespace/newlines
        sig_match = re.match(
            r'\s*pub\s+unsafe\s+extern\s+"C"\s+fn\s+(\w+)\s*\(',
            after_attr
        )

        if not sig_match:
            # Not a matching function, keep as-is
            result.append('#[no_mangle]')
            pos = idx + len('#[no_mangle]')
            continue

        func_name = sig_match.group(1)

        if func_name in excludes:
            # Keep excluded functions as-is
            result.append('#[no_mangle]')
            pos = idx + len('#[no_mangle]')
            continue

        # Find the opening paren of args
        paren_start = idx + len('#[no_mangle]') + sig_match.end() - 1

        # Find matching closing paren (handle nested parens/generics)
        paren_depth = 1
        paren_end = paren_start + 1
        while paren_end < len(content) and paren_depth > 0:
            if content[paren_end] == '(':
                paren_depth += 1
            elif content[paren_end] == ')':
                paren_depth -= 1
            paren_end += 1
        paren_end -= 1  # points to closing )

        args_str = content[paren_start + 1:paren_end]

        # Find return type (between ) and {)
        after_paren = content[paren_end + 1:]
        ret_match = re.match(r'\s*->\s*(.+?)\s*\{', after_paren, re.DOTALL)
        void_match = re.match(r'\s*\{', after_paren) if not ret_match else None

        if not ret_match and not void_match:
            # Unexpected syntax
            result.append('#[no_mangle]')
            pos = idx + len('#[no_mangle]')
            continue

        is_void = void_match is not None
        if is_void:
            ret_type = None
            brace_start = paren_end + 1 + void_match.end() - 1
        else:
            ret_type = ret_match.group(1).strip()
            brace_start = paren_end + 1 + ret_match.end() - 1

        # Find matching closing brace
        brace_end = find_matching_brace(content, brace_start)
        if brace_end == -1:
            result.append('#[no_mangle]')
            pos = idx + len('#[no_mangle]')
            continue

        # Extract the function body
        body = content[brace_start + 1:brace_end]

        # Parse argument names only
        _, arg_names = parse_args(args_str)

        # Get the indentation
        line_start = content.rfind('\n', 0, idx) + 1
        indent = ''
        for ch in content[line_start:idx]:
            if ch in ' \t':
                indent += ch
            else:
                break

        # Generate the _impl function + wrapper
        if is_void:
            impl_fn = f"{indent}unsafe fn {func_name}_impl({args_str}) {{{body}}}\n\n"
            wrapper_fn = (
                f"{indent}#[no_mangle]\n"
                f"{indent}pub unsafe extern \"C\" fn {func_name}({args_str}) {{\n"
                f"{indent}    rgpu_common::ffi::catch_panic((), || {func_name}_impl({arg_names}))\n"
                f"{indent}}}"
            )
        else:
            impl_fn = f"{indent}unsafe fn {func_name}_impl({args_str}) -> {ret_type} {{{body}}}\n\n"
            wrapper_fn = (
                f"{indent}#[no_mangle]\n"
                f"{indent}pub unsafe extern \"C\" fn {func_name}({args_str}) -> {ret_type} {{\n"
                f"{indent}    rgpu_common::ffi::catch_panic({error_value}, || {func_name}_impl({arg_names}))\n"
                f"{indent}}}"
            )

        result.append(impl_fn)
        result.append(wrapper_fn)
        pos = brace_end + 1

    return ''.join(result)


def transform_bare_ffi(content, error_value, excludes=None):
    """Transform bare `unsafe extern "C" fn` (no #[no_mangle]) that haven't been transformed yet."""
    if excludes is None:
        excludes = set()

    result = []
    pos = 0

    while pos < len(content):
        # Look for bare `unsafe extern "C" fn` (not preceded by #[no_mangle] or already _impl)
        match = re.search(r'\nunsafe extern "C" fn (\w+)\s*\(', content[pos:])
        if not match:
            result.append(content[pos:])
            break

        func_name = match.group(1)
        abs_pos = pos + match.start() + 1  # +1 for the \n

        # Skip if already transformed (_impl suffix) or excluded
        if func_name.endswith('_impl') or func_name in excludes:
            result.append(content[pos:abs_pos + len(match.group(0)) - 1])
            pos = abs_pos + len(match.group(0)) - 1
            continue

        # Check it's not preceded by #[no_mangle] (already handled)
        line_start = content.rfind('\n', 0, abs_pos - 1)
        prev_lines = content[max(0, line_start - 50):abs_pos].strip()
        if '#[no_mangle]' in prev_lines:
            result.append(content[pos:abs_pos + len(match.group(0)) - 1])
            pos = abs_pos + len(match.group(0)) - 1
            continue

        # Find opening paren
        paren_start_idx = content.index('(', abs_pos)
        paren_depth = 1
        paren_end = paren_start_idx + 1
        while paren_end < len(content) and paren_depth > 0:
            if content[paren_end] == '(':
                paren_depth += 1
            elif content[paren_end] == ')':
                paren_depth -= 1
            paren_end += 1
        paren_end -= 1

        args_str = content[paren_start_idx + 1:paren_end]

        # Find return type or void
        after_paren = content[paren_end + 1:]
        ret_match = re.match(r'\s*->\s*(.+?)\s*\{', after_paren, re.DOTALL)
        void_match = re.match(r'\s*\{', after_paren) if not ret_match else None

        if not ret_match and not void_match:
            result.append(content[pos:abs_pos + len(match.group(0)) - 1])
            pos = abs_pos + len(match.group(0)) - 1
            continue

        is_void = void_match is not None
        if is_void:
            ret_type = None
            brace_start = paren_end + 1 + void_match.end() - 1
        else:
            ret_type = ret_match.group(1).strip()
            brace_start = paren_end + 1 + ret_match.end() - 1

        brace_end = find_matching_brace(content, brace_start)
        if brace_end == -1:
            result.append(content[pos:abs_pos + len(match.group(0)) - 1])
            pos = abs_pos + len(match.group(0)) - 1
            continue

        body = content[brace_start + 1:brace_end]
        _, arg_names = parse_args(args_str)

        # Generate transformed code
        result.append(content[pos:abs_pos])

        if is_void:
            impl_fn = f"unsafe fn {func_name}_impl({args_str}) {{{body}}}\n\n"
            wrapper_fn = (
                f"unsafe extern \"C\" fn {func_name}({args_str}) {{\n"
                f"    rgpu_common::ffi::catch_panic((), || {func_name}_impl({arg_names}))\n"
                f"}}"
            )
        else:
            impl_fn = f"unsafe fn {func_name}_impl({args_str}) -> {ret_type} {{{body}}}\n\n"
            wrapper_fn = (
                f"unsafe extern \"C\" fn {func_name}({args_str}) -> {ret_type} {{\n"
                f"    rgpu_common::ffi::catch_panic({error_value}, || {func_name}_impl({arg_names}))\n"
                f"}}"
            )

        result.append(impl_fn)
        result.append(wrapper_fn)
        pos = brace_end + 1

    return ''.join(result)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_file> <error_value> [--exclude=func1,func2] [--also-bare]")
        sys.exit(1)

    input_file = sys.argv[1]
    error_value = sys.argv[2]
    excludes = set()
    also_bare = False

    for arg in sys.argv[3:]:
        if arg.startswith('--exclude='):
            excludes = set(arg[len('--exclude='):].split(','))
        elif arg == '--also-bare':
            also_bare = True

    output = transform_file(input_file, error_value, excludes)

    if also_bare:
        output = transform_bare_ffi(output, error_value, excludes)

    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"Transformed: {input_file}")
