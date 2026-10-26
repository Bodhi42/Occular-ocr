"""Параллельная обработка страниц процессами (occular.parallel)."""
import sys
import types

import pytest

import occular.parallel as P


def test_auto_workers_bounds(monkeypatch):
    monkeypatch.setattr(P, "available_cpus", lambda: 128)
    monkeypatch.setattr(P, "_available_memory_gb", lambda: 1000.0)
    assert P.auto_workers(3) == 3                       # не больше задач
    assert P.auto_workers(2) == 1                       # 2 страницы — без пула (запуск дороже)
    assert P.auto_workers(500) == P.AUTO_MAX_WORKERS    # авто-потолок
    monkeypatch.setattr(P, "_available_memory_gb", lambda: 7.0)
    assert P.auto_workers(500) == int(7.0 // P.GB_PER_WORKER)   # по памяти
    monkeypatch.setattr(P, "available_cpus", lambda: 2)
    monkeypatch.setattr(P, "_available_memory_gb", lambda: None)
    assert P.auto_workers(500) == 2                     # по ядрам


def test_processes_safe_requires_main_guard(tmp_path, monkeypatch):
    guarded = tmp_path / "a.py"; guarded.write_text('if __name__ == "__main__":\n    run()\n')
    plain = tmp_path / "b.py"; plain.write_text("run()\n")
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__=str(guarded)))
    assert P.processes_safe()
    monkeypatch.setattr(P, "_warned", False)
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__=str(plain)))
    with pytest.warns(RuntimeWarning, match="__main__"):
        assert not P.processes_safe()
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace())   # REPL / Jupyter
    assert P.processes_safe()


def test_can_use_processes():
    ok = types.SimpleNamespace(_init_kwargs=dict(detector_name="dbnet-onnx", recognizer_name="crnn-onnx",
                                                 unwarp=None, unwarp_gate=True))
    import occular  # noqa: F401 — регистрирует стандартные компоненты
    assert P.can_use_processes(ok)
    assert not P.can_use_processes(types.SimpleNamespace(_init_kwargs={**ok._init_kwargs, "unwarp": object()}))
    assert not P.can_use_processes(types.SimpleNamespace(_init_kwargs={**ok._init_kwargs, "unwarp_gate": lambda a, b: True}))
    assert not P.can_use_processes(types.SimpleNamespace())


def test_pdf_pages_in_processes_match_sequential(tmp_path):
    fitz = pytest.importorskip("fitz")
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "sample_page.png"
    if not sample.exists():
        pytest.skip("нет sample_page.png")
    pdf = tmp_path / "three.pdf"
    doc = fitz.open()
    for _ in range(3):
        img = fitz.open(str(sample)); r = img[0].rect
        page = doc.new_page(width=r.width * 72 / 200, height=r.height * 72 / 200)
        page.insert_image(page.rect, filename=str(sample))
    doc.save(str(pdf))
    from occular import OCRPipeline, Settings
    try:
        p = OCRPipeline(Settings(lm=False, num_threads=2))
    except Exception as e:
        pytest.skip(f"модели недоступны: {e}")
    if not P.processes_safe():
        pytest.skip("главный модуль без __main__-защиты — процессы не запускаются")
    seq = p.process_pdf(str(pdf), workers=1)
    par = p.process_pdf(str(pdf), workers=2)
    assert [pg["page"] for pg in par] == [1, 2, 3]
    assert [[l["text"] for l in pg["results"]] for pg in par] == [[l["text"] for l in pg["results"]] for pg in seq]


def test_init_worker_keeps_quiet_stream_open(monkeypatch):
    # логгер, заведённый при построении конвейера в воркере, держит подменённый stderr:
    # после инициализации запись в него не должна падать «I/O operation on closed file»
    import logging
    import occular.pipeline as pipeline_mod
    held = {}

    class FakePipeline:
        def __init__(self, **kw):
            held["handler"] = logging.StreamHandler()          # берёт текущий sys.stderr

    monkeypatch.setattr(pipeline_mod, "OCRPipeline", FakePipeline)
    monkeypatch.setattr(P, "_PIPE", None, raising=False)
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    import occular._runtime as rt
    monkeypatch.setattr(rt, "LOW_MEMORY", False)
    err = sys.stderr
    P._init_worker({}, 1)
    assert sys.stderr is err
    held["handler"].stream.write("late warning\n")             # не должно бросать ValueError
    held["handler"].stream.flush()


def test_threads_per_worker_split_physical_cores(monkeypatch):
    monkeypatch.setattr(P, "physical_cpus", lambda: 12)  # напр. Apple M: 12 ядер без SMT
    assert P.threads_per_worker(1, 8) == 8                 # не больше запрошенного
    assert P.threads_per_worker(6, 8) == 2
    assert P.threads_per_worker(24, 8) == 1                # минимум 1


def test_physical_cpus_fallback_without_topology(monkeypatch):
    monkeypatch.setattr(P.sys, "platform", "win32")        # ОС не сообщает топологию
    for logical, physical in ((16, 8), (4, 4)):
        monkeypatch.setattr(P, "_PHYSICAL", None)
        monkeypatch.setattr(P, "available_cpus", lambda n=logical: n)
        assert P.physical_cpus() == physical


def test_physical_cpus_real_machine(monkeypatch):
    monkeypatch.setattr(P, "_PHYSICAL", None)
    assert 1 <= P.physical_cpus() <= P.available_cpus()
