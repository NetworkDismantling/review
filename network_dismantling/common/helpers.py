from pathlib import Path


def extend_filename(file: Path, filename_extension: str, postfixes=None) -> Path:
    suffixes = []

    file_split = file.stem.split(".")
    file_name = file_split[0]

    if len(file_split) > 1:
        suffixes.extend(file_split[1:])

    if postfixes is not None:
        if isinstance(postfixes, str):
            postfixes = [postfixes]

        suffixes.extend(postfixes)

    if len(file.suffix) > 0:
        suffixes.append(file.suffix.replace(".", ""))

    extended_file = file.with_name(file_name + filename_extension)

    if len(suffixes) > 0:
        suffix = "." + ".".join(suffixes)
        extended_file = extended_file.with_suffix(suffix)

    return extended_file
