"""Run the shipped tests from the built source distribution."""
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archives = sorted(Path('dist').glob('*.tar.gz'))
if len(archives) != 1:
    raise SystemExit('Build exactly one source distribution before this check')
with tempfile.TemporaryDirectory(prefix='omniread-sdist-') as directory:
    with tarfile.open(archives[0]) as archive:
        archive.extractall(directory, filter='data')
    roots = list(Path(directory).iterdir())
    if len(roots) != 1:
        raise SystemExit('Unexpected source distribution layout')
    root = roots[0]
    for name in ['tests/conftest.py', 'tests/fixtures/corpus.json']:
        if not (root / name).is_file():
            raise SystemExit(f'Source distribution is missing {name}')
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root, check=True)
