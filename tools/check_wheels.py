"""Проверка перед релизом: не придётся ли pip что-то КОМПИЛИРОВАТЬ при `pip install occular-ocr`.

pip берёт новейшую версию зависимости в пределах ограничений. Если у неё нет колеса под систему
пользователя, но есть исходники (sdist), pip не откатывается на старую версию с колесом, а собирает
новую из исходников (так opencv 4.11 собирался ~45 мин на M1 с macOS 12, issue #4).

Скрипт читает requirements.txt, для каждой платформы и версии Python вычисляет маркеры и повторяет
выбор pip по данным PyPI. Печатает BUILD (будет компиляция) и NO DIST (pip не найдёт ничего —
платформа не поддерживается, установка упадёт сразу, без компиляции: pip собирает пакеты только после
разрешения всех зависимостей). Код выхода 1, если на поддерживаемой платформе будет компиляция, или
если в маркерах есть platform_release/platform_version (ломают pip на Linux).

    python tools/check_wheels.py            # нужны packaging и доступ к pypi.org
"""
import json
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename
from packaging.version import Version

PYTHONS = [(3, v) for v in range(8, 16)]


def _manylinux(arch, glibc):
    out = [f"manylinux_2_{m}_{arch}" for m in range(glibc, 16, -1)] + [f"manylinux2014_{arch}"]
    if arch == "x86_64":
        out += ["manylinux2010_x86_64", "manylinux1_x86_64"]
    return out


# (имя, теги платформы, окружение маркеров PEP 508, поддерживается ли). glibc 2.28 — минимум onnxruntime.
# macOS 10.15 не поддерживается: onnxruntime под неё не собран.
PLATFORMS = [
    ("macOS 11 Apple silicon", list(mac_platforms((11, 0), "arm64")), dict(sys_platform="darwin", platform_machine="arm64", platform_release="20.6.0"), True),
    ("macOS 12 Apple silicon", list(mac_platforms((12, 0), "arm64")), dict(sys_platform="darwin", platform_machine="arm64", platform_release="21.6.0"), True),
    ("macOS 13 Apple silicon", list(mac_platforms((13, 0), "arm64")), dict(sys_platform="darwin", platform_machine="arm64", platform_release="22.6.0"), True),
    ("macOS 14 Apple silicon", list(mac_platforms((14, 0), "arm64")), dict(sys_platform="darwin", platform_machine="arm64", platform_release="23.6.0"), True),
    ("macOS 15 Apple silicon", list(mac_platforms((15, 0), "arm64")), dict(sys_platform="darwin", platform_machine="arm64", platform_release="24.6.0"), True),
    ("macOS 10.15 Intel", list(mac_platforms((10, 15), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="19.6.0"), False),
    ("macOS 11 Intel", list(mac_platforms((11, 0), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="20.6.0"), True),
    ("macOS 12 Intel", list(mac_platforms((12, 0), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="21.6.0"), True),
    ("macOS 13 Intel", list(mac_platforms((13, 0), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="22.6.0"), True),
    ("macOS 14 Intel", list(mac_platforms((14, 0), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="23.6.0"), True),
    ("macOS 15 Intel", list(mac_platforms((15, 0), "x86_64")), dict(sys_platform="darwin", platform_machine="x86_64", platform_release="24.6.0"), True),
    ("Linux x86_64 glibc 2.28", _manylinux("x86_64", 28), dict(sys_platform="linux", platform_machine="x86_64"), True),
    ("Linux x86_64 glibc 2.35", _manylinux("x86_64", 35), dict(sys_platform="linux", platform_machine="x86_64"), True),
    ("Linux aarch64 glibc 2.28", _manylinux("aarch64", 28), dict(sys_platform="linux", platform_machine="aarch64"), True),
    ("Linux aarch64 glibc 2.35", _manylinux("aarch64", 35), dict(sys_platform="linux", platform_machine="aarch64"), True),
    ("Windows x64", ["win_amd64"], dict(sys_platform="win32", platform_machine="AMD64"), True),
    ("Windows ARM64", ["win_arm64"], dict(sys_platform="win32", platform_machine="ARM64"), False),
]


@lru_cache(None)
def releases(name):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=60) as r:
        data = json.load(r)
    out = {}
    for v, files in data["releases"].items():
        try:
            V = Version(v)
        except Exception:
            continue
        files = [f for f in files if not f.get("yanked")]
        if files and not V.is_prerelease:
            out[V] = files
    return out


def pip_choice(req, py, plats):
    """'wheel' | 'BUILD <v>' | 'NO DIST' — что сделает pip без --only-binary."""
    tags = set(cpython_tags(py, platforms=plats)) | set(compatible_tags(py, platforms=plats))
    pyv = Version(f"{py[0]}.{py[1]}")
    for V in sorted((v for v in releases(req.name) if v in req.specifier), reverse=True):
        files = [f for f in releases(req.name)[V]
                 if not f.get("requires_python") or pyv in SpecifierSet(f["requires_python"])]
        if any(f["filename"].endswith(".whl") and set(parse_wheel_filename(f["filename"])[3]) & tags for f in files):
            return "wheel"
        if any(f["packagetype"] == "sdist" for f in files):
            return f"BUILD {V}"
    return "NO DIST"


def main():
    reqs = [Requirement(l) for l in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    bad = 0
    # platform_release/platform_version — свободные строки ОС («6.8.0-138-generic» на Linux), а не версии:
    # сравнение «<»/«>=» с ними pip < 26 роняет с InvalidVersion на всех системах, а не только на целевой.
    for r in reqs:
        if r.marker and any(k in str(r.marker) for k in ("platform_release", "platform_version")):
            print(f"ЗАПРЕЩЁННЫЙ МАРКЕР в {r}: platform_release/platform_version ломают pip на Linux")
            bad += 1
    for pname, plats, env, supported in PLATFORMS:
        for py in PYTHONS:
            e = dict(env, python_version=f"{py[0]}.{py[1]}", python_full_version=f"{py[0]}.{py[1]}.0",
                     os_name="nt" if env["sys_platform"] == "win32" else "posix", implementation_name="cpython",
                     platform_python_implementation="CPython", platform_system={"darwin": "Darwin", "linux": "Linux",
                                                                                "win32": "Windows"}[env["sys_platform"]])
            issues = []
            active = {}                           # одно имя в нескольких строках — pip пересекает ограничения
            for r in reqs:
                if r.marker and not r.marker.evaluate(e):
                    continue
                if r.name in active:
                    active[r.name].specifier &= r.specifier
                else:
                    active[r.name] = Requirement(f"{r.name}{r.specifier}")
            for r in active.values():
                c = pip_choice(r, py, plats)
                if c != "wheel":
                    issues.append(f"{r.name}: {c}")
            if issues:
                # pip компилирует только после того, как разрешил ВСЕ зависимости: если чего-то нет совсем
                # (NO DIST), установка падает сразу, до сборки (у исходников лишь читаются метаданные)
                build = any("BUILD" in i for i in issues) and not any("NO DIST" in i for i in issues)
                bad += build and supported
                tag = "" if supported else " (не поддерживается)"
                print(f"{pname}{tag}, Python {py[0]}.{py[1]}: " + "; ".join(issues))
    print("OK: на поддерживаемых платформах ничего не компилируется" if not bad else f"ПРОБЛЕМ: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
