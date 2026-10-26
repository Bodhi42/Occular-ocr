"""Кладёт C++-рантайм Microsoft (5 DLL) в occular/_vcrt для колеса win_amd64.

Источник — официальный vc_redist.x64.exe: внутри него cab-архивы, файлы лежат как msvcp140.dll_amd64 и т.п.
Распаковка — cabextract (apt install cabextract). Хэши файлов закреплены ниже: если Microsoft обновил
redistributable, скрипт остановится и покажет новые хэши — проверить версию и обновить PINNED + NOTICE.txt.

    python tools/fetch_vcrt.py                    # скачать с aka.ms
    python tools/fetch_vcrt.py --exe vc_redist.x64.exe
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://aka.ms/vc14/vc_redist.x64.exe"
VERSION = "14.51.36247.0"
PINNED = {   # sha256 файлов из vc_redist.x64.exe 14.51.36247.0
    "vcruntime140.dll": "d1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7",
    "vcruntime140_1.dll": "a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e",
    "msvcp140.dll": "7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca",
    "msvcp140_1.dll": "206c931bf90fdad8816de3b5e2ef80b2bcaa9406c89ecc05fe6fddffe251e982",
    "msvcp140_atomic_wait.dll": "3d0cbfaa1bf3eecf5a3f4491d2960ee803cb994f30292c6adc4a07c498f60e2b",
}
DEST = Path(__file__).resolve().parent.parent / "occular" / "_vcrt"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", help="готовый vc_redist.x64.exe (иначе скачать с aka.ms)")
    a = ap.parse_args()
    if not shutil.which("cabextract"):
        sys.exit("нужен cabextract: apt install cabextract")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        exe = Path(a.exe) if a.exe else t / "vc_redist.x64.exe"
        if not a.exe:
            print(f"скачиваю {URL}")
            urllib.request.urlretrieve(URL, exe)
        # внешний exe — бандл WiX с приклеенными cab; cabextract находит их сканированием
        subprocess.run(["cabextract", "-q", "-d", str(t / "outer"), str(exe)], capture_output=True)
        found = {}
        for cab in sorted((t / "outer").iterdir()):
            listing = subprocess.run(["cabextract", "-l", str(cab)], capture_output=True, text=True).stdout
            if "msvcp140.dll_amd64" in listing:
                subprocess.run(["cabextract", "-q", "-d", str(t / "x"), str(cab)], capture_output=True)
                for name in PINNED:
                    f = t / "x" / f"{name}_amd64"
                    if f.exists():
                        found[name] = f
                break
        missing = set(PINNED) - set(found)
        if missing:
            sys.exit(f"в {exe.name} не нашлись: {', '.join(sorted(missing))}")
        wrong = {n: sha256(f) for n, f in found.items() if sha256(f) != PINNED[n]}
        if wrong:
            print(f"хэши не совпали с закреплёнными (версия {VERSION}) — Microsoft обновил redistributable?")
            for n, h in wrong.items():
                print(f'    "{n}": "{h}",')
            sys.exit("проверьте версию файлов и обновите PINNED и occular/_vcrt/NOTICE.txt")
        DEST.mkdir(exist_ok=True)
        for name, f in found.items():
            shutil.copyfile(f, DEST / name)
        print(f"готово: {len(found)} DLL ({VERSION}) → {DEST}")


if __name__ == "__main__":
    main()
