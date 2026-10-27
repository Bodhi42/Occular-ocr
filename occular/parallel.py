"""Параллельная обработка многих страниц пулом ПРОЦЕССОВ (PDF, папка в CLI).

Почему процессы, а не потоки: постобработка детектора, раскладка строк и чисто-питоновский декод LM
держат GIL, поэтому потоки одного процесса упираются в одно ядро. Замер (EPYC 7763, 64 ядра,
64 страницы): потоки — потолок ~2.2 стр/с с нативным декодером, а с чисто-питоновским скорость
ПАДАЕТ после 8 потоков (0.66 → 0.44 стр/с на 128); отдельные процессы — 3.8 стр/с на 64.

Каждый воркер один раз строит свой конвейер (≈2.5 с: модели + LM) и берёт страницы из общей очереди
по одной — тяжёлые и лёгкие страницы не простаивают друг за другом. Цена — память: модели и LM в
каждом процессе, пик ≈0.75 ГБ на воркер (арена ONNX Runtime в воркерах выключена); авто-число
воркеров ограничено свободной памятью.
"""
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

GB_PER_WORKER = 1.0          # с запасом к пику RSS ≈0.8 ГБ на воркер (замер, 64 стр. PDF): арена ONNX
                             # Runtime в воркерах выключена и память возвращается после каждой страницы
                             # (с ареной было ≈1.95 ГБ); из них ≈0.26 ГБ — общие библиотеки
AUTO_MAX_WORKERS = 16        # авто-потолок; больше — только явным workers=N
THREAD_WORKERS_MAX = 8       # потоков-воркеров, когда процессы нельзя: с чисто-питоновским декодером
                             # 8 потоков — 0.66 стр/с, 16 — 0.54, 128 — 0.44 (GIL)

_PIPE = None                 # конвейер внутри процесса-воркера


def available_cpus() -> int:
    """Логические CPU, доступные процессу (с учётом affinity/cgroup, где ОС это сообщает)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _linux_physical(cpus) -> Optional[int]:
    """Физические ядра среди доступных процессу CPU: уникальные (сокет, ядро) из sysfs."""
    cores = set()
    for c in cpus:
        base = f"/sys/devices/system/cpu/cpu{c}/topology/"
        try:
            with open(base + "physical_package_id") as f1, open(base + "core_id") as f2:
                cores.add((f1.read().strip(), f2.read().strip()))
        except OSError:
            return None
    return len(cores) or None


def _macos_physical() -> Optional[int]:
    """hw.physicalcpu: у Apple Silicon равно числу логических (SMT нет), у Intel-маков — половина."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        val, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(b"hw.physicalcpu", ctypes.byref(val), ctypes.byref(size), None, 0) == 0:
            return val.value or None
    except (OSError, AttributeError):
        pass
    return None


_PHYSICAL: Optional[int] = None


def physical_cpus() -> int:
    """Физические ядра, доступные процессу. Linux — по топологии из sysfs (в пределах affinity),
    macOS — sysctl hw.physicalcpu; где ОС не сообщает (Windows), считаем, что от 8 логических CPU
    есть SMT и ядер вдвое меньше."""
    global _PHYSICAL
    if _PHYSICAL is None:
        cpus = available_cpus()
        found = None
        if sys.platform.startswith("linux"):
            try:
                found = _linux_physical(sorted(os.sched_getaffinity(0)))
            except (AttributeError, OSError):
                found = None
        elif sys.platform == "darwin":
            found = _macos_physical()
        _PHYSICAL = max(1, min(cpus, found)) if found else (cpus // 2 if cpus >= 8 else cpus)
    return _PHYSICAL


def threads_per_worker(workers: int, wanted: int) -> int:
    """Потоков ONNX Runtime на воркер: делим ФИЗИЧЕСКИЕ ядра. Замеры: EPYC (SMT), 512 стр. —
    64 воркера × 2 потока (все 128 логических) 2.66 стр/с, 64 × 1 (только физические) 3.18;
    Apple M (12 ядер без SMT), 32 стр. — 6 воркеров × 2 потока 2.42 стр/с, 6 × 1 — 2.19."""
    return max(1, min(wanted, physical_cpus() // max(1, workers)))


def _available_memory_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (AttributeError, ValueError, OSError):
        return None                                      # macOS/Windows без psutil — не знаем


AUTO_MIN_TASKS = 3           # меньше страниц — без пула: запуск воркера ≈2.5 с, а страница в этом
                             # процессе (8 потоков) ≈1.6 с — 2 страницы последовательно быстрее


def auto_workers(n_tasks: int, threads_per_worker: int = 1) -> int:
    """Сколько процессов брать по умолчанию: по ядрам, по свободной памяти, не больше задач и
    AUTO_MAX_WORKERS (на больших машинах больше — явным workers=N); меньше AUTO_MIN_TASKS задач — 1."""
    if n_tasks < AUTO_MIN_TASKS:
        return 1
    n = min(n_tasks, max(1, available_cpus() // max(1, threads_per_worker)), AUTO_MAX_WORKERS)
    mem = _available_memory_gb()
    if mem is not None:
        n = min(n, max(1, int(mem // GB_PER_WORKER)))
    return max(1, n)


_warned = False


def processes_safe() -> bool:
    """Можно ли запускать процессы-воркеры. Они стартуют через spawn (на Windows и macOS иначе нельзя,
    а fork после старта ONNX Runtime/OpenCV зависает): новый процесс заново импортирует главный
    скрипт пользователя. Если в скрипте нет `if __name__ == "__main__":`, каждый воркер повторил бы
    весь OCR скрипта — поэтому без этой защиты остаёмся на потоках и один раз подсказываем.
    REPL/Jupyter (у главного модуля нет файла) — безопасно; внутри воркера — нельзя."""
    global _warned
    import multiprocessing as mp
    if mp.parent_process() is not None:
        return False
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path:
        return True
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            guarded = re.search(r"""if\s+__name__\s*==\s*['"]__main__['"]""", f.read()) is not None
    except OSError:
        guarded = False
    if not guarded and not _warned:
        _warned = True
        warnings.warn("occular: pages are processed by threads, not processes, because the main script has "
                      "no `if __name__ == \"__main__\":` guard. With the guard, multi-page work scales "
                      "across many cores (processes are not limited by the GIL).", RuntimeWarning, stacklevel=3)
    return guarded


def can_use_processes(pipeline) -> bool:
    """Пул процессов воспроизводит конвейер по аргументам конструктора. Нельзя, если детектор или
    распознаватель зарегистрирован вне occular (в новом процессе его не будет)."""
    kw = getattr(pipeline, "_init_kwargs", None)
    if kw is None:
        return False
    from .registry import Registry
    for name, table, default in ((kw.get("detector_name"), Registry._detectors, Registry._default_detector),
                                 (kw.get("recognizer_name"), Registry._recognizers, Registry._default_recognizer)):
        cls = table.get(name or default)
        if cls is None or not cls.__module__.startswith("occular."):
            return False
    return True


def _init_worker(kwargs: dict, threads: int):
    # до построения конвейера: пул потоков нативного декодера (rayon) — по числу потоков воркера,
    # иначе каждый процесс заводит пул на все ядра машины
    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
    global _PIPE
    from . import _runtime
    _runtime.LOW_MEMORY = True                            # без арены ONNX Runtime: память ×0.45, время +4 %
    from .pipeline import OCRPipeline
    # «Loaded … detector» не печатать N раз. /dev/null не закрываем: логгеры, созданные при построении
    # (huggingface_hub заводит StreamHandler на текущий sys.stderr), держат его до конца процесса,
    # и запись в закрытый файл давала «ValueError: I/O operation on closed file».
    err, sys.stderr = sys.stderr, open(os.devnull, "w")
    try:
        _PIPE = OCRPipeline(**{**kwargs, "num_threads": threads})
        # LM грузится лениво — на первой странице, уже после восстановления stderr, и её строка
        # печаталась по разу на воркер. Задачи в пуле — только страницы под OCR, LM нужна всё равно.
        rec = getattr(_PIPE, "recognizer", None)
        if getattr(rec, "lm", False) and hasattr(rec, "_ensure_lm"):
            rec._ensure_lm()
    finally:
        sys.stderr = err


_TRIM = None


def _trim():
    """Вернуть ОС освобождённую память malloc (glibc): иначе воркер от страницы к странице растёт
    из-за фрагментации — замер на 64 стр. PDF: пик 1081 МБ без, 819 МБ с возвратом после страницы."""
    global _TRIM
    if _TRIM is None:
        try:
            import ctypes
            _TRIM = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _TRIM = False                                 # не glibc (macOS, Windows, musl) — пропускаем
    if _TRIM:
        _TRIM(0)


def _run_task(task: Tuple):
    try:
        return _run_task_inner(task)
    finally:
        _trim()


def _run_task_inner(task: Tuple):
    kind = task[0]
    if kind == "image":
        res = _PIPE.process_image(task[1])
        return list(res), getattr(res, "tables", None)
    if kind == "pdf":                                     # ("pdf", path, page_num, dpi, force_ocr)
        from .pipeline import pymupdf
        fitz = pymupdf()
        _, path, page_num, dpi, force_ocr = task
        with fitz.open(path) as doc:
            return _PIPE._process_pdf_page(doc, page_num, dpi, force_ocr)
    raise ValueError(kind)


def run_tasks(pipeline, tasks: Sequence[Tuple], workers: int, threads_per_worker: int = 1,
              on_done=None) -> List:
    """Выполнить задачи ("image", path) / ("pdf", path, page, dpi, force_ocr) в workers процессах.
    Результаты — в порядке задач; исключение задачи кладётся на её место (вызывающий решает).
    on_done(i, result) — по мере готовности (прогресс, запись файлов)."""
    ctx = get_context("spawn")                            # fork после старта ONNX Runtime небезопасен
    out: List = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(pipeline._init_kwargs, threads_per_worker)) as ex:
        futs = {ex.submit(_run_task, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:                       # noqa: BLE001 — отдаём вызывающему
                out[i] = e
            if on_done is not None:
                on_done(i, out[i])
    return out
