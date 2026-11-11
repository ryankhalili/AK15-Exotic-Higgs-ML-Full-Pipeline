"""Audit publishable source paths, private literals and accidental binary artifacts."""
import argparse
import json
from pathlib import Path
import re
import subprocess

IGNORED = {'.git', '.venv', '.work', 'work', 'inputs', 'external', '__pycache__', '.pytest_cache'}
SUFFIXES = {'.py', '.C', '.cc', '.h', '.xml', '.json', '.md', '.txt', '.ini', '.sh'}
PATTERNS = {
    'personal_windows_path': re.compile(r'(?i)(?<![A-Za-z])[A-Z]:[\\/]'),
    'private_storage_path': re.compile(r'/(?:afs|nfs_scratch|hdfs|store/user)/'),
    'personal_home_path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'credential': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,})'),
    'fine_grained_token': re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
    'url_credentials': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
    'dated_campaign': re.compile(r'(?<!\d)20[2-9]\d[01]\d[0-3]\d(?!\d)'),
}


def scan(root, forbidden=(), exact_paths=None):
    files, failures = [], []
    candidates = sorted(root.rglob('*')) if exact_paths is None else [root / path for path in exact_paths]
    for path in candidates:
        relative = path.relative_to(root)
        if set(relative.parts) & IGNORED:
            if exact_paths is not None:
                failures.append(dict(file=relative.as_posix(), reason='excluded artifact in publication set'))
            continue
        if '..' in relative.parts or relative.is_absolute():
            failures.append(dict(file=str(relative), reason='unsafe relative path'))
            continue
        if path.is_symlink():
            failures.append(dict(file=relative.as_posix(), reason='symlink'))
            continue
        if not path.is_file():
            continue
        files.append(relative.as_posix())
        if (path.suffix not in SUFFIXES and path.name not in {'.gitignore', '.gitattributes'}) or path.name == 'AGENTS.md':
            failures.append(dict(file=relative.as_posix(), reason='unexpected artifact type'))
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeError:
            failures.append(dict(file=relative.as_posix(), reason='binary/non-UTF8 content'))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                failures.append(dict(file=relative.as_posix(), reason=label))
        for literal in forbidden:
            if literal.casefold() in content.casefold():
                failures.append(dict(file=relative.as_posix(), reason='forbidden literal'))
    return files, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbid', action='append', default=[], help='Additional literal; not stored in output')
    parser.add_argument('--inventory', action='store_true', help='Write relative publishable paths to FILES.txt')
    parser.add_argument('--staged', action='store_true', help='Audit exactly the Git index; excluded directories are errors')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exact = None
    if args.staged:
        raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root)
        exact = [Path(name) for name in raw.decode('utf-8').split('\0') if name]
        if not exact:
            raise ValueError('Empty Git index')
        for path in exact:
            staged = subprocess.check_output(['git', 'show', ':' + path.as_posix()], cwd=root)
            if staged.replace(b'\r\n', b'\n') != (root / path).read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError('Staged and working bytes differ: ' + path.as_posix())
    files, failures = scan(root, args.forbid, exact_paths=exact)
    if args.inventory and not failures:
        (root/'FILES.txt').write_text('\n'.join(sorted(set(files) | {'FILES.txt'})) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=len(files), failures=failures, excluded_local_directories=sorted(IGNORED)), indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
