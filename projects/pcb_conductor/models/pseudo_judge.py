import numpy as np
import cv2

def pseudo_label_judge_24pins_no_equal_spacing(
    p: np.ndarray,                 # HxW prob in [0,1]
    thr: float = 0.5,
    min_area: int = 30,
    bottom_band_ratio: float = 0.12,
    target_n: int = 24,
    n_tol: int = 4,                # 允许 24±4（断裂/粘连时更鲁棒）
):
    """
    Returns:
      s_img: float in [0,1]
      m_rel: HxW uint8 reliable pixels
      info: dict metrics
    """
    H, W = p.shape
    y = (p >= thr).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(y, connectivity=8)

    comps = []
    for i in range(1, num):
        x, y0, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        comps.append(dict(i=i, x=x, y=y0, w=w, h=h, area=area, cx=cx, cy=cy))

    n_cc = len(comps)
    if n_cc == 0:
        m_rel = (np.abs(p - 0.5) > 0.25).astype(np.uint8)
        return 0.0, m_rel, dict(n_cc=0, reason="no_components")

    ws  = np.array([c["w"] for c in comps], dtype=np.float32)
    hs  = np.array([c["h"] for c in comps], dtype=np.float32)
    cxs = np.array([c["cx"] for c in comps], dtype=np.float32)

    # ---- verticality / shape ----
    aspect = hs / (ws + 1e-6)
    good_aspect_ratio = float(np.mean(aspect > 2.0))  # pins tall
    w_cv = float(np.std(ws) / (np.mean(ws) + 1e-6))   # width consistency

    # ---- "merged pins" proxy: abnormally wide components ----
    # use robust baseline: median width
    med_w = float(np.median(ws))
    wide_ratio = float(np.mean(ws > 1.8 * med_w))  # too wide => likely merged

    # ---- bottom connection check (critical) ----
    b = max(int(H * bottom_band_ratio), 1)
    bottom = y[H - b : H, :]
    bottom_fill = float(np.mean(bottom))
    # also check if bottom has a long horizontal run (optional stronger)
    # bottom_run = float(np.max(np.sum(bottom, axis=1)) / W)

    # ---- spacing sanity (NOT equal spacing) ----
    cxs_sorted = np.sort(cxs)
    if len(cxs_sorted) >= 3:
        dx = np.diff(cxs_sorted)
        med_dx = float(np.median(dx) + 1e-6)
        gap_outlier = (dx < 0.3 * med_dx) | (dx > 3.0 * med_dx)
        gap_outlier_ratio = float(np.mean(gap_outlier))
    else:
        med_dx = 0.0
        gap_outlier_ratio = 1.0

    # ---- confidence proxy ----
    conf = float(np.mean(np.abs(p - 0.5) * 2.0))  # 0..1

    # ---- scoring ----
    # count prior: soft around 24
    count_err = abs(n_cc - target_n)
    s_count = float(np.clip(1.0 - (count_err / (n_tol + 1e-6)), 0.0, 1.0))

    # bottom should be low (tune these two numbers to your crop)
    s_bottom = float(np.clip((0.22 - bottom_fill) / (0.22 - 0.12 + 1e-6), 0.0, 1.0))

    # width consistency: smaller w_cv better
    s_wcons = float(np.clip((0.50 - w_cv) / (0.50 - 0.20 + 1e-6), 0.0, 1.0))

    # verticality: want most comps tall
    s_aspect = float(np.clip((good_aspect_ratio - 0.5) / (0.9 - 0.5 + 1e-6), 0.0, 1.0))

    # merged penalty: fewer wide comps is better
    s_merge = float(np.clip(1.0 - wide_ratio, 0.0, 1.0))

    # spacing sanity: only penalize extreme outliers, not non-uniformity
    s_gap = float(np.clip(1.0 - gap_outlier_ratio, 0.0, 1.0))

    # combine (count + bottom + merge most important for your case)
    s_img = (
        0.30 * s_count +
        0.20 * s_bottom +
        0.20 * s_merge +
        0.15 * s_wcons +
        0.10 * s_aspect +
        0.03 * s_gap +
        0.02 * conf
    )
    s_img = float(np.clip(s_img, 0.0, 1.0))

    # ---- reliable pixel mask ----
    # high-confidence pixels
    m_conf = (np.abs(p - 0.5) > 0.20).astype(np.uint8)

    # keep only pixels that belong to kept components if positive
    keep = np.zeros((H, W), dtype=np.uint8)
    kept_ids = {c["i"] for c in comps}
    for i in kept_ids:
        keep[labels == i] = 1

    y_bin = (p >= thr).astype(np.uint8)
    m_rel = (m_conf & ((1 - y_bin) | keep)).astype(np.uint8)

    info = dict(
        n_cc=n_cc,
        s_count=s_count,
        good_aspect_ratio=good_aspect_ratio,
        w_cv=w_cv,
        wide_ratio=wide_ratio,
        bottom_fill=bottom_fill,
        gap_outlier_ratio=gap_outlier_ratio,
        conf=conf,
        s_img=s_img,
        med_w=med_w,
        med_dx=med_dx,
    )
    return s_img, m_rel, info
