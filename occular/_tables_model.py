"""Модель структуры таблиц (FastTab, split[+merge]) — определение сети для CPU-torch пути.
Скопировано как самодостаточный модуль (зависит только от torch). Используется occular/tables.py."""
import torch, torch.nn as nn, torch.nn.functional as F


def conv_gn(cin, cout, k=3, s=1, groups=8):
    if isinstance(s, int): s = (s, s)
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k//2, bias=False),
                         nn.GroupNorm(groups, cout), nn.ReLU(inplace=True))


class ResBlk(nn.Module):
    def __init__(s, c):
        super().__init__(); s.a = conv_gn(c, c); s.b = conv_gn(c, c)
    def forward(s, x): return x + s.b(s.a(x))


class Encoder(nn.Module):
    """Fully-conv. Горизонталь всегда /8. Вертикаль /16 (vdown=16) или /8 (vdown=8, лучше плотные строки)."""
    def __init__(s, dmodel=256, vdown=16):
        super().__init__()
        ch = [32, 64, 128, dmodel]
        s.s1 = nn.Sequential(conv_gn(3, ch[0], 3, (2, 2)), ResBlk(ch[0]))       # /2 /2
        s.s2 = nn.Sequential(conv_gn(ch[0], ch[1], 3, (2, 2)), ResBlk(ch[1]))   # /4 /4
        s.s3 = nn.Sequential(conv_gn(ch[1], ch[2], 3, (2, 2)), ResBlk(ch[2]))   # /8 /8
        s4stride = (2, 1) if vdown == 16 else (1, 1)                            # 4-я стадия: только вертикаль или none
        s.s4 = nn.Sequential(conv_gn(ch[2], ch[3], 3, s4stride), ResBlk(ch[3]))
    def forward(s, x): return s.s4(s.s3(s.s2(s.s1(x))))                         # (B, dmodel, Hf, W/8)


class TRM(nn.Module):
    def __init__(s, dmodel, T=6):
        super().__init__(); s.T = T; s.ln = nn.LayerNorm(dmodel)
        s.psi = nn.Sequential(nn.Linear(2*dmodel, dmodel), nn.GELU(), nn.Linear(dmodel, dmodel))
    def forward(s, F_map):
        g = F_map.mean(dim=(-2, -1)); z = torch.zeros_like(g)
        for _ in range(s.T):
            z = z + s.psi(torch.cat([s.ln(z), g], dim=-1))
        return z                                            # (B, dmodel) табличный контекст (T=0 -> нули)


class AxisHead(nn.Module):
    """Одна ось: 1D-последовательность (уже свёрнута по перпенд.оси) -> tiny-Transformer -> счётчик + границы."""
    def __init__(s, cin, zdim, dseq=256, nmax=85, L=2, A=4, dff=512):
        super().__init__()
        s.proj = nn.Conv1d(cin, dseq, 1); s.zcond = nn.Linear(zdim, dseq)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dseq, A, dff, batch_first=True, activation='gelu'), L) if L > 0 else None
        s.smooth = nn.Sequential(nn.Conv1d(dseq, dseq, 3, 1, 1, groups=dseq), nn.Conv1d(dseq, dseq, 1))
        s.count = nn.Linear(dseq, nmax); s.interval = nn.Linear(dseq, nmax+1)
    def forward(s, seq, z):
        h = s.proj(seq) + s.zcond(z).unsqueeze(-1)
        if s.tf is not None: h = s.tf(h.transpose(1, 2)).transpose(1, 2)
        h = s.smooth(h); pooled = h.mean(dim=-1)
        cb = torch.cumsum(F.softmax(s.interval(pooled), dim=-1), dim=-1)
        return s.count(pooled), cb


class AxisHeadAttn(nn.Module):
    """Одна ось, но границы читаются ПРОСТРАНСТВЕННО, а не из усреднённого вектора.

    Мотивация (замерено): в AxisHead пространственная ось сворачивается mean-пулом, и один Linear должен
    восстановить все nmax+1 длин интервалов из ОДНОГО вектора. Тест переобучения на 100 таблицах даёт
    struct-F1 0.766 НА ТРЕНЕ при медианной ошибке границы 21.6 px (шаг фичемапа 16 px) и росте ошибки
    с числом строк (14.7 px при <=10 строках -> 85.5 px при 35) => сквозь пул позиция не проходит.
    Это же объясняет, почему oracle-карта сепараторов давала +0.0016: сигнал упирался в пул, а не был бесполезен.

    Здесь: nmax+1 обучаемых запросов (по одному на слот интервала) читают последовательность через attention
    (+ обучаемое позиционное кодирование — без него attention не может локализовать). Границы по-прежнему
    МОНОТОННЫ ПО ПОСТРОЕНИЮ (cumsum softmax), порогов нет, ops только ONNX-дружественные (matmul/softmax).
    Счётчик остаётся из пула: oracle-тесты показали, что счётчик НЕ боттлнек, а его природа глобальная."""
    def __init__(s, cin, zdim, dseq=256, nmax=85, L=2, A=4, dff=512, maxlen=512):
        super().__init__()
        s.proj = nn.Conv1d(cin, dseq, 1); s.zcond = nn.Linear(zdim, dseq)
        s.pos = nn.Parameter(torch.zeros(maxlen, dseq)); nn.init.normal_(s.pos, std=0.02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dseq, A, dff, batch_first=True, activation='gelu'), L) if L > 0 else None
        s.smooth = nn.Sequential(nn.Conv1d(dseq, dseq, 3, 1, 1, groups=dseq), nn.Conv1d(dseq, dseq, 1))
        s.q = nn.Parameter(torch.zeros(nmax+1, dseq)); nn.init.normal_(s.q, std=0.02)
        s.ln = nn.LayerNorm(dseq); s.ival = nn.Linear(dseq, 1); s.count = nn.Linear(dseq, nmax)
        s.scale = dseq ** -0.5
    def forward(s, seq, z):
        h = s.proj(seq) + s.zcond(z).unsqueeze(-1)          # [B,dseq,Lp]
        Lp = h.shape[-1]
        h = h + s.pos[:Lp].t().unsqueeze(0)                 # позиционное кодирование (иначе позиция не читается)
        if s.tf is not None: h = s.tf(h.transpose(1, 2)).transpose(1, 2)
        h = s.smooth(h)
        hT = s.ln(h.transpose(1, 2))                        # [B,Lp,dseq]
        att = torch.softmax(torch.matmul(s.q.unsqueeze(0), hT.transpose(1, 2))*s.scale, dim=-1)  # [B,nmax+1,Lp]
        ctx = torch.matmul(att, hT)                         # [B,nmax+1,dseq] — свой пространственный обзор на слот
        cb = torch.cumsum(torch.softmax(s.ival(ctx).squeeze(-1), dim=-1), dim=-1)
        return s.count(h.mean(dim=-1)), cb


class AxisHeadDense(nn.Module):
    """Одна ось: ПЛОТНАЯ регрессия индекса полосы вместо восстановления длин из пула.

    Почему так (замерено, тест переобучения на 100 таблицах):
      * pooled-голова (AxisHead): struct-F1 НА ТРЕНЕ 0.766 — не может заучить даже 100 таблиц => упирается
        не объём данных, а выразительность: mean-пул уничтожает ГДЕ границы, и один Linear должен
        восстановить nmax+1 длин из одного вектора. Ошибка растёт с числом строк (14.7 -> 85.5 px).
      * attention-голова (AxisHeadAttn): 0.601 — ХУЖЕ. Замена одного способа сжатия другим не лечит.

    Формулировка здесь: на КАЖДОЙ позиции i предсказываем неотрицательный прирост d_i (softplus), затем
    phi = cumsum(d) — монотонно неубывающая «дробный индекс полосы» вдоль оси. Таргет phi_gt(y) = кусочно-линейная
    интерполяция: phi_gt(Y_k) = k. Обучаем L1 на phi (плотный сигнал в КАЖДОЙ позиции, а не 86 чисел из вектора).
    Границы на инференсе = phi^{-1}(k) (обращение монотонной функции линейной интерполяцией) — субпиксельная
    точность не ограничена шагом фичемапа. Счётчик = phi(конец) — согласован с границами по построению, отдельная
    CE-голова не нужна (но оставлена для совместимости лосса).
    Монотонность — ПО ПОСТРОЕНИЮ (cumsum неотрицательного). Ops только ONNX-дружественные."""
    def __init__(s, cin, zdim, dseq=256, nmax=85, L=2, A=4, dff=512, maxlen=512):
        super().__init__()
        s.nmax = nmax
        s.proj = nn.Conv1d(cin, dseq, 1); s.zcond = nn.Linear(zdim, dseq)
        s.pos = nn.Parameter(torch.zeros(maxlen, dseq)); nn.init.normal_(s.pos, std=0.02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dseq, A, dff, batch_first=True, activation='gelu'), L) if L > 0 else None
        s.smooth = nn.Sequential(nn.Conv1d(dseq, dseq, 3, 1, 1, groups=dseq), nn.Conv1d(dseq, dseq, 1))
        s.dens = nn.Conv1d(dseq, 1, 1)                      # плотность полос на позицию
        s.count = nn.Linear(dseq, nmax)                     # доп. счётчик (совместимость с CE-лоссом)
    def forward(s, seq, z):
        h = s.proj(seq) + s.zcond(z).unsqueeze(-1)
        Lp = h.shape[-1]
        h = h + s.pos[:Lp].t().unsqueeze(0)
        if s.tf is not None: h = s.tf(h.transpose(1, 2)).transpose(1, 2)
        h = s.smooth(h)
        d = F.softplus(s.dens(h).squeeze(1))                # [B,Lp] >= 0
        phi = torch.cumsum(d, dim=-1)                       # монотонно неубывающая
        return s.count(h.mean(dim=-1)), phi


def phi_to_bounds(phi, nmax):
    """phi [B,Lp] (монотонна) -> нормированные cumsum-границы [B,nmax+1] в [0,1], как ждёт метрика.
    Граница k = phi^{-1}(k)/Lp через линейную интерполяцию; за пределами счёта паддинг 1.0 (как cumbounds)."""
    B, Lp = phi.shape
    dev = phi.device
    out = torch.ones(B, nmax+1, device=dev)
    R = phi[:, -1].clamp(min=1e-6)                          # предсказанное число полос (дробное)
    for b in range(B):
        p = phi[b]; n = int(torch.round(R[b]).item())   # round, НЕ floor: phi_конец=2.9 при 3 строках теряло целую строку (ошибка 126 px на малых таблицах)
        n = max(1, min(nmax, n))
        ks = torch.arange(1, n+1, device=dev, dtype=p.dtype)
        idx = torch.searchsorted(p.contiguous(), ks.contiguous())
        idx = idx.clamp(1, Lp-1)
        p0 = p[idx-1]; p1 = p[idx]
        frac = (ks-p0)/(p1-p0).clamp(min=1e-6)
        pos = (idx.to(p.dtype)-1+frac.clamp(0, 1)+1.0)/Lp   # центр ячейки -> нормировка
        out[b, :n] = pos.clamp(0, 1)
    return out


class AxisHeadSeg(nn.Module):
    """Одна ось: КЛАССИФИКАЦИЯ КАЖДОЙ ПОЗИЦИИ «это сепаратор?» — каноническая формулировка split-подхода.

    Литература (SPLERGE/ICDAR19, RobusTabNet, SEM, TABLET) решает split именно так: классификация по каждому
    пикселю/позиции вдоль оси (BCE на heatmap сепараторов), а НЕ регрессия длин интервалов из сжатого вектора.
    TABLET отдельно отмечает мотив: отказ от предсказания боксов снижает потерю разрешения.
    Сам FastTab (arXiv:2605.22422) тоже предсказывает СЕПАРАТОРЫ, а не длины из пула — наша прежняя
    реализация отклонилась от статьи, на которую ссылается.

    Наши замеры, обосновывающие переход (тест переобучения на 100 таблицах, struct-F1 НА ТРЕНЕ):
      pooled (регрессия длин из mean-пула) 0.766 | attention-запросы 0.601 | dense-phi + вес у границ 0.804
    Все три упираются в то, что позиция границы проходит через сжатие. Здесь сжатия нет вообще:
    выход — логит на позицию, декодирование = центры масс связных компонент (порог 0.5), без обучаемого счётчика.
    ONNX-дружественно (conv/sigmoid); декодирование — на numpy вне графа, как в SPLERGE."""
    def __init__(s, cin, zdim, dseq=256, nmax=85, L=2, A=4, dff=512, maxlen=512):
        super().__init__()
        s.nmax = nmax
        s.proj = nn.Conv1d(cin, dseq, 1); s.zcond = nn.Linear(zdim, dseq)
        s.pos = nn.Parameter(torch.zeros(maxlen, dseq)); nn.init.normal_(s.pos, std=0.02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dseq, A, dff, batch_first=True, activation='gelu'), L) if L > 0 else None
        s.smooth = nn.Sequential(nn.Conv1d(dseq, dseq, 3, 1, 1, groups=dseq), nn.Conv1d(dseq, dseq, 1))
        s.sep = nn.Conv1d(dseq, 1, 1)                       # логит «сепаратор» на КАЖДУЮ позицию
        s.count = nn.Linear(dseq, nmax)                     # вспомогательный счётчик (не используется в декоде)
    def forward(s, seq, z):
        h = s.proj(seq) + s.zcond(z).unsqueeze(-1)
        Lp = h.shape[-1]
        h = h + s.pos[:Lp].t().unsqueeze(0)
        if s.tf is not None: h = s.tf(h.transpose(1, 2)).transpose(1, 2)
        h = s.smooth(h)
        return s.count(h.mean(dim=-1)), s.sep(h).squeeze(1)  # [B,nmax], [B,Lp] логиты


def sep_to_bounds(logit, nmax):
    """Логиты сепараторов [B,Lp] -> нормированные cumsum-границы [B,nmax+1] (как ждёт метрика).
    Декод как в SPLERGE: порог 0.5 -> связные компоненты -> центр масс каждой = граница. Порог обоснован тем,
    что это классификация (в отличие от cumsum-формулировки, где порогов не было по построению)."""
    B, Lp = logit.shape
    prob = torch.sigmoid(logit)
    out = torch.ones(B, nmax+1, device=logit.device)
    for b in range(B):
        p = prob[b]; on = (p > 0.5)
        cent = []
        i = 0
        while i < Lp:
            if on[i]:
                j = i
                while j+1 < Lp and on[j+1]: j += 1
                w = p[i:j+1]; idx = torch.arange(i, j+1, device=p.device, dtype=p.dtype)
                cent.append(float(((w*idx).sum()/w.sum().clamp(min=1e-6)+0.5)/Lp))
                i = j+1
            else: i += 1
        cent = [c for c in cent if 1e-4 < c < 1-1e-4][:nmax-1]      # внутренние границы
        n = len(cent)+1                                            # полос = внутренних границ + 1
        if n > 1: out[b, :n-1] = torch.tensor(cent, device=logit.device)
        out[b, n-1:] = 1.0
    return out


class MergeHead(nn.Module):
    """MERGE-фаза: классификация РЁБЕР между соседними клетками канон-сетки.

    Формулировка канонична для split->merge подхода (SPLERGE/SEM): сначала сетка, потом объединения.
    Признак ребра — конкатенация фич ДВУХ соседних клеток, взятых из карты бэкбона в центрах клеток
    через grid_sample (та же карта, что у split-голов -> накладные расходы +3.7% ко времени шага,
    замерено; повторный проход бэкбона давал бы +74%).

    Почему рёбра, а не (rowspan, colspan) на клетку: ребро — локальное бинарное решение, инвариантное
    к размеру таблицы, а логические ячейки восстанавливаются union-find'ом (merge_target.cells_from_edges,
    round-trip против OTSL = 1.0000). Предсказание спанов напрямую страдает от несогласованности соседей.

    ONNX: только conv1x1, grid_sample, cat, linear, gelu — все экспортируемы (grid_sample с opset>=16).
    """

    def __init__(s, cin, d=128, bedge=False):
        super().__init__()
        s.proj = nn.Conv2d(cin, d, 1)
        s.bedge = bedge                                     # сэмплировать признак НА ГРАНИЦЕ между клетками
        s.mlp = nn.Sequential(nn.Linear(d*(3 if bedge else 2), d), nn.GELU(), nn.Linear(d, 1))

    def forward(s, Fm, rcb, ccb, R, C):
        """Fm: (B,Cin,Hf,Wf); rcb/ccb: нормированные КУМУЛЯТИВНЫЕ границы (B,Rmax)/(B,Cmax);
        R,C: (B,) число полос. Возвращает список (hor, ver) на элемент — размеры разные по батчу."""
        P = s.proj(Fm)
        out = []
        for b in range(P.shape[0]):
            r = int(R[b]); c = int(C[b])
            if r < 1 or c < 1:
                out.append((None, None)); continue
            y = torch.cat([torch.zeros(1, device=P.device), rcb[b, :r]])
            x = torch.cat([torch.zeros(1, device=P.device), ccb[b, :c]])
            cy = (y[:-1] + y[1:]) * 0.5                      # центры полос строк, (r,)
            cx = (x[:-1] + x[1:]) * 0.5                      # центры полос столбцов, (c,)
            gy = cy.view(r, 1).expand(r, c); gx = cx.view(1, c).expand(r, c)
            grid = torch.stack([gx*2-1, gy*2-1], -1).unsqueeze(0)         # (1,r,c,2)
            f = F.grid_sample(P[b:b+1], grid, align_corners=False)[0].permute(1, 2, 0)  # (r,c,d)
            if not s.bedge:
                hor = s.mlp(torch.cat([f[:, :-1], f[:, 1:]], -1)).squeeze(-1) if c > 1 else None
                ver = s.mlp(torch.cat([f[:-1], f[1:]], -1)).squeeze(-1) if r > 1 else None
            else:
                # РЕШАЮЩАЯ УЛИКА ЛЕЖИТ НА ГРАНИЦЕ, А НЕ В ЦЕНТРАХ: слить две клетки или нет —
                # это вопрос, есть ли между ними линия разлиновки. Признаки в двух ЦЕНТРАХ этой
                # линии не видят вовсе, голова вынуждена её домысливать. Сэмплируем ещё и границу.
                hor = ver = None
                if c > 1:                                    # граница между (i,j) и (i,j+1) -> x = x[j+1]
                    bx = x[1:-1].view(1, c-1).expand(r, c-1)
                    by = cy.view(r, 1).expand(r, c-1)
                    gb = torch.stack([bx*2-1, by*2-1], -1).unsqueeze(0)
                    fb = F.grid_sample(P[b:b+1], gb, align_corners=False)[0].permute(1, 2, 0)
                    hor = s.mlp(torch.cat([f[:, :-1], f[:, 1:], fb], -1)).squeeze(-1)
                if r > 1:                                    # граница между (i,j) и (i+1,j) -> y = y[i+1]
                    by = y[1:-1].view(r-1, 1).expand(r-1, c)
                    bx = cx.view(1, c).expand(r-1, c)
                    gb = torch.stack([bx*2-1, by*2-1], -1).unsqueeze(0)
                    fb = F.grid_sample(P[b:b+1], gb, align_corners=False)[0].permute(1, 2, 0)
                    ver = s.mlp(torch.cat([f[:-1], f[1:], fb], -1)).squeeze(-1)
            out.append((hor, ver))
        return out



class MergeHeadGrid(nn.Module):
    """Merge-голова по канону литературы (RobusTabNet Grid CNN, arXiv:2203.09056 §3.3.3).

    Три отличия от MergeHead, каждое взято из статьи:

    1. РЕГИОННЫЙ ПУЛИНГ ВМЕСТО ТОЧКИ. RobusTabNet берёт RoI Align 7x7 по боксу КАЖДОЙ ячейки, а не
       признак в одной точке. Одна точка в центре не знает ни размера клетки, ни что внутри неё
       (текст? пусто?) — а пустота клетки как раз главный признак объединения.
       Здесь: adaptive_avg_pool по боксу клетки на карте Fm (дешевле RoI Align, тот же смысл).

    2. КОНТЕКСТ ПО СЕТКЕ (это и есть «Grid CNN»). Признаки клеток укладываются в карту
       F_grid (M,N,d) и прогоняются через 3 свёртки 3x3: «to aggregate context information
       effectively with several stacked convolution layers to improve cell merging accuracy».
       Без этого решение о ребре принимается локально, без ведома соседних рёбер, и получается
       несогласованная разметка (у нас: recall 0.40 при precision 0.72).

    3. ПРИЗНАК ПРОСТРАНСТВЕННОЙ СОВМЕСТИМОСТИ l_ij. Relation network в статье склеивает
       x_ij = [F_i; l_ij; F_j], где l_ij кодирует геометрию пары. Голая пара признаков не знает,
       насколько клетки разного размера и далеко ли друг от друга.

    Плюс СОБСТВЕННОЕ дополнение (не из статьи): признак НА ГРАНИЦЕ между клетками — решение
    «слить или нет» зависит от наличия линии разлиновки, которую центры клеток не видят.
    """

    def __init__(s, cin, d=128, ncv=3, bedge=True):
        super().__init__()
        s.proj = nn.Conv2d(cin, d, 1)
        s.bedge = bedge
        s.gcnn = nn.Sequential(*[m for _ in range(ncv) for m in
                                 (nn.Conv2d(d, d, 3, 1, 1), nn.GroupNorm(8, d), nn.GELU())])
        nrel = d*2 + (d if bedge else 0) + 5          # +5 = l_ij (геометрия пары)
        s.mlp = nn.Sequential(nn.Linear(nrel, d), nn.GELU(), nn.Linear(d, 1))

    def _pool_cells(s, P, y, x, r, c):
        """Признак каждой клетки = среднее по её боксу на карте P (регионный пулинг)."""
        _, dd, Hf, Wf = P.shape
        yi = (y*Hf).clamp(0, Hf); xi = (x*Wf).clamp(0, Wf)
        out = P.new_zeros(r, c, dd)
        for i in range(r):
            a = int(yi[i].floor()); b = max(a+1, int(yi[i+1].ceil())); b = min(b, Hf)
            if a >= b: a = max(0, b-1)
            for j in range(c):
                u = int(xi[j].floor()); v = max(u+1, int(xi[j+1].ceil())); v = min(v, Wf)
                if u >= v: u = max(0, v-1)
                out[i, j] = P[0, :, a:b, u:v].mean(dim=(-2, -1))
        return out

    def forward(s, Fm, rcb, ccb, R, C):
        P0 = s.proj(Fm)
        out = []
        for b in range(P0.shape[0]):
            r = int(R[b]); c = int(C[b])
            if r < 1 or c < 1:
                out.append((None, None)); continue
            y = torch.cat([torch.zeros(1, device=P0.device), rcb[b, :r]])
            x = torch.cat([torch.zeros(1, device=P0.device), ccb[b, :c]])
            f = s._pool_cells(P0[b:b+1], y, x, r, c)                  # (r,c,d) регионный пулинг
            g = s.gcnn(f.permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0)   # Grid CNN: контекст
            hw = (y[1:]-y[:-1]).view(r, 1).expand(r, c)               # высоты клеток
            ww = (x[1:]-x[:-1]).view(1, c).expand(r, c)               # ширины клеток
            cy = ((y[:-1]+y[1:])*0.5).view(r, 1).expand(r, c)
            cx = ((x[:-1]+x[1:])*0.5).view(1, c).expand(r, c)
            hor = ver = None
            if c > 1:
                lij = torch.stack([ww[:, :-1], ww[:, 1:], hw[:, :-1],
                                   (cx[:, 1:]-cx[:, :-1]), (cy[:, 1:]-cy[:, :-1])], -1)
                parts = [g[:, :-1], g[:, 1:]]
                if s.bedge:
                    bx = x[1:-1].view(1, c-1).expand(r, c-1); by = cy[:, :-1]
                    gb = torch.stack([bx*2-1, by*2-1], -1).unsqueeze(0)
                    parts.append(F.grid_sample(P0[b:b+1], gb, align_corners=False)[0].permute(1, 2, 0))
                hor = s.mlp(torch.cat(parts + [lij], -1)).squeeze(-1)
            if r > 1:
                lij = torch.stack([hw[:-1], hw[1:], ww[:-1],
                                   (cy[1:]-cy[:-1]), (cx[1:]-cx[:-1])], -1)
                parts = [g[:-1], g[1:]]
                if s.bedge:
                    by = y[1:-1].view(r-1, 1).expand(r-1, c); bx = cx[:-1]
                    gb = torch.stack([bx*2-1, by*2-1], -1).unsqueeze(0)
                    parts.append(F.grid_sample(P0[b:b+1], gb, align_corners=False)[0].permute(1, 2, 0))
                ver = s.mlp(torch.cat(parts + [lij], -1)).squeeze(-1)
            out.append((hor, ver))
        return out

class DirBlock(nn.Module):
    """Направленный dilated depthwise-блок: горизонтальные ядра (1×k) для строк-сепараторов ИЛИ вертикальные (k×1) для столбцов."""
    def __init__(s, c, k=9, dil=1, vertical=False):
        super().__init__()
        ksz = (k, 1) if vertical else (1, k); pad = (dil*(k//2), 0) if vertical else (0, dil*(k//2)); dl = (dil, 1) if vertical else (1, dil)
        s.dw = nn.Conv2d(c, c, ksz, 1, pad, dilation=dl, groups=c, bias=False); s.bn1 = nn.BatchNorm2d(c)
        s.pw = nn.Conv2d(c, c, 1, bias=False); s.bn2 = nn.BatchNorm2d(c)
    def forward(s, x):
        r = x; x = F.silu(s.bn1(s.dw(x))); x = s.bn2(s.pw(x)); return F.silu(x+r)


class SeparatorHead(nn.Module):
    """Плотные 2D-карты сепараторов (строки/столбцы) из фичемапа. Только ONNX-ops. Карты в mapres×mapres."""
    def __init__(s, cin=256, c=64, mapres=96):
        super().__init__(); s.mapres = mapres
        s.stem = nn.Sequential(nn.Conv2d(cin, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        s.rowb = nn.Sequential(DirBlock(c, 9, 1, False), DirBlock(c, 9, 2, False), DirBlock(c, 9, 4, False))   # 1×9 -> строки
        s.colb = nn.Sequential(DirBlock(c, 9, 1, True), DirBlock(c, 9, 2, True), DirBlock(c, 9, 4, True))      # 9×1 -> столбцы
        s.rout = nn.Conv2d(c, 1, 1); s.cout = nn.Conv2d(c, 1, 1)
    def forward(s, Fm):
        h = s.stem(Fm)
        rm = F.interpolate(s.rout(s.rowb(h)), size=(s.mapres, s.mapres), mode='bilinear', align_corners=False)
        cm = F.interpolate(s.cout(s.colb(h)), size=(s.mapres, s.mapres), mode='bilinear', align_corners=False)
        return rm, cm                                       # логиты [B,1,mapres,mapres]


class FastTabSplit(nn.Module):
    def __init__(s, dmodel=256, dseq=256, Rmax=85, Cmax=52, T=6, L=2, vdown=16, pool='mean', sep_prior=False, mapres=96, head='pooled', upax=1, merge=False, bedge=False, mgrid=False):
        super().__init__()
        s.pool = pool; s.sep_prior = sep_prior; s.mapres = mapres; s.head = head; s.upax = upax; s.upax_col = 0   # 0 = как upax
        s.merge = merge
        s.enc = Encoder(dmodel, vdown); s.trm = TRM(dmodel, T)
        cin = dmodel*(2 if pool == 'meanmax' else 1)
        HD = {'attn': AxisHeadAttn, 'dense': AxisHeadDense, 'seg': AxisHeadSeg}.get(head, AxisHead)   # dense = плотная регрессия индекса
        # 'hybrid' (замерено на тесте потолка): dense выигрывает по СТОЛБЦАМ (0.935 против 0.796),
        # pooled — по СТРОКАМ (0.735 против 0.69). Берём для каждой оси свою лучшую голову.
        RH = AxisHead if head == 'hybrid' else HD
        CH = AxisHeadDense if head == 'hybrid' else HD
        s.row = RH(cin, dmodel, dseq, Rmax, L); s.col = CH(cin, dmodel, dseq, Cmax, L)
        if merge: s.mrg = (MergeHeadGrid(dmodel, bedge=bedge) if mgrid
                          else MergeHead(dmodel, bedge=bedge))   # см. MergeHead / MergeHeadGrid
        if sep_prior:                                       # голова + инъекция max-профиля (gate zero-init -> старт идентичен)
            s.sep = SeparatorHead(dmodel, 64, mapres)
            s.erow = nn.Conv1d(1, cin, 5, 1, 2); s.ecol = nn.Conv1d(1, cin, 5, 1, 2)
            s.alpha_row = nn.Parameter(torch.zeros(1)); s.alpha_col = nn.Parameter(torch.zeros(1))
    def _seq(s, Fm, axis):
        if s.pool == 'meanmax': q = torch.cat([Fm.mean(dim=axis), Fm.amax(dim=axis)], 1)
        else: q = Fm.mean(dim=axis)
        up = s.upax
        if axis == -2 and getattr(s, 'upax_col', 0): up = s.upax_col     # столбцам можно своё разрешение оси
        if up > 1: q = F.interpolate(q, scale_factor=up, mode='linear', align_corners=False)
        return q                                            # dense-голове нужно слотов >> nmax (иначе границы не разрешить)
    def forward(s, x, gate_off=False, override=None):        # override=(prow_map,pcol_map) вероятности [B,1,mapres,mapres] для oracle-тестов
        Fm = s.enc(x); z = s.trm(Fm)
        rseq = s._seq(Fm, -1); cseq = s._seq(Fm, -2)        # [B,cin,Hf], [B,cin,Wf]
        out = {}
        if s.sep_prior:
            rm, cm = s.sep(Fm); out['rmap'] = rm; out['cmap'] = cm
            pcol = override[1] if override is not None else torch.sigmoid(cm)
            prow = override[0] if override is not None else torch.sigmoid(rm)
            q_col = pcol.amax(dim=2)                         # max по высоте -> [B,1,mapres(ширина)]
            q_row = F.avg_pool1d(prow.amax(dim=3), s.mapres//rseq.shape[-1])   # max по ширине, пул до Hf
            gc = 0.0 if gate_off else torch.tanh(s.alpha_col); gr = 0.0 if gate_off else torch.tanh(s.alpha_row)
            cseq = cseq + gc*s.ecol(q_col); rseq = rseq + gr*s.erow(q_row)
        rcnt, rcb = s.row(rseq, z); ccnt, ccb = s.col(cseq, z)
        if s.head == 'dense':                               # rcb/ccb здесь = phi; конвертируем для метрики
            out.update(rphi=rcb, cphi=ccb)
            out.update(rcnt=rcnt, rcb=phi_to_bounds(rcb, s.row.nmax), ccnt=ccnt, ccb=phi_to_bounds(ccb, s.col.nmax))
            return out
        if s.head == 'seg':                                 # rcb/ccb здесь = логиты сепараторов на позицию
            out.update(rsep=rcb, csep=ccb)
            rb = sep_to_bounds(rcb, s.row.nmax); cb = sep_to_bounds(ccb, s.col.nmax)
            out.update(rcnt=rcnt, rcb=rb, ccnt=ccnt, ccb=cb)
            if s.merge:                                     # рёбра сетки на ТОЙ ЖЕ карте Fm (+3.7% времени)
                R = torch.argmax(rcnt, dim=-1) + 1; C = torch.argmax(ccnt, dim=-1) + 1
                out['medges'] = s.mrg(Fm, rb, cb, R, C)
            return out
        if s.head == 'hybrid':                              # строки как в mainline, столбцы через phi
            out.update(cphi=ccb)
            out.update(rcnt=rcnt, rcb=rcb, ccnt=ccnt, ccb=phi_to_bounds(ccb, s.col.nmax))
            return out
        out.update(rcnt=rcnt, rcb=rcb, ccnt=ccnt, ccb=ccb); return out


