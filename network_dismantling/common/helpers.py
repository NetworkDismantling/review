from pathlib import Path


def extend_filename(file: Path, filename_extension: str, postfixes=None) -> Path:
    suffixes = []

    file_split = file.stem.split('.')
    file_name = file_split[0]

    if len(file_split) > 1:
        suffixes.extend(file_split[1:])

    if postfixes is not None:
        if isinstance(postfixes, str):
            postfixes = [postfixes]

        suffixes.extend(postfixes)

    suffixes.append(file.suffix.replace(".", ""))
    suffix = '.' + '.'.join(suffixes)

    extended_file = file.with_name(file_name + filename_extension).with_suffix(suffix)

    return extended_file
