import os
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

from data import load_csv, numeric_columns, prepare_matrix
from geom import (radar_points, raster_mask, overlap_uniqueness, team_discrete_stats,
                     pack_mask, unpack_mask, save_masks_cache, load_masks_cache)
from optimize import random_groups, hillclimb
import model as nn_model

st.set_page_config(page_title="Diverse Group Matcher", layout="wide")

DATA_DEFAULT = "Data-Table 1.csv"

# -------------------------------
# Geometric mask caching
# -------------------------------
MASK_CACHE_DIR = ".mask_cache"
MASK_CACHE_VERSION = "v1"  # bump if mask generation logic changes

def _mask_cache_path(data_path: str, feature_cols, grid_n: int) -> str:
    abs_path = os.path.abspath(data_path)
    try:
        stat = os.stat(abs_path)
        mtime = int(stat.st_mtime)
        size = int(stat.st_size)
    except Exception:
        mtime, size = 0, 0

    cols_key = ",".join([str(c) for c in feature_cols])
    key = f"{MASK_CACHE_VERSION}|{abs_path}|{mtime}|{size}|grid={int(grid_n)}|cols={cols_key}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()

    os.makedirs(MASK_CACHE_DIR, exist_ok=True)
    return os.path.join(MASK_CACHE_DIR, f"masks_{h}.npz")


@st.cache_data(show_spinner=False, max_entries=50)
def _load_masks_from_disk(cache_path: str, n_items: int, grid_n: int):
    """Loads masks from disk cache if present and compatible; returns None otherwise."""
    if not os.path.exists(cache_path):
        return None
    try:
        packed_masks, areas, cached_grid_n, _span = load_masks_cache(cache_path)
        if int(cached_grid_n) != int(grid_n):
            return None
        if int(packed_masks.shape[0]) != int(n_items):
            return None

        shape = (int(grid_n), int(grid_n))
        masks = [unpack_mask(packed_masks[i], shape) for i in range(int(n_items))]
        return masks
    except Exception:
        # If cache is corrupt/partial, ignore and rebuild.
        return None


def _save_masks_to_disk(cache_path: str, masks, grid_n: int, span: float = 1.05) -> None:
    """Saves masks to disk cache in a compact packed-bits format."""
    packed = np.stack([pack_mask(m) for m in masks], axis=0)
    areas = np.array([int(m.sum()) for m in masks], dtype=np.int32)
    save_masks_cache(cache_path, packed, areas, int(grid_n), float(span))


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return load_csv(path)


@st.cache_data(show_spinner=False, max_entries=10)
def compute_masks(Xn: np.ndarray, grid_n: int = 140, span: float = 1.05):
    """Precompute discrete raster masks for every item's radar polygon."""
    d = Xn.shape[1]
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False)
    masks = []
    for i in range(Xn.shape[0]):
        poly = radar_points(Xn[i], angles)
        masks.append(raster_mask(poly, grid_n=grid_n, span=span))
    return masks


def make_exact_pair_scorer(masks):
    """Exact uniqueness score = 1 - Jaccard overlap of polygon area masks."""
    cache = {}

    def score(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        if key in cache:
            return cache[key]
        s = overlap_uniqueness(masks[key[0]], masks[key[1]])
        cache[key] = float(s)
        return float(s)

    return score


def main():
    st.title("Diverse Group Matcher (Discrete Geometry + Keras)")
    st.caption(
        "Each item becomes a multi-axis shape (radar polygon). We match items/groups to maximize non-overlapping area "
        "(uniqueness) while optionally balancing variance across groups."
    )


    # --- About / Instructions ---
    st.markdown(
        '''
<div style="padding:16px 18px;border-radius:14px;border:1px solid rgba(148,163,184,0.35);background:rgba(15,23,42,0.08);">
  <div style="font-size:20px;font-weight:700;margin-bottom:6px;color:rgba(255,255,255,0.98);">How to use this app</div>
  <div style="font-size:13px;line-height:1.45;color:rgba(255,255,255,0.95);">
    <ol>
      <li><b>Choose your CSV</b> in the left sidebar (default: <code>Data-Table 1.csv</code>).</li>
      <li><b>Select numeric feature columns</b> (axes) that define each item’s geometry.</li>
      <li>Pick a mode:
        <ul>
          <li><b>Group mode</b>: choose <i>items per group</i> + <i>number of groups</i>.</li>
          <li><b>Pair-only mode</b>: forms disjoint pairs (group size fixed to 2).</li>
        </ul>
      </li>
      <li>Optionally enable <b>Keras acceleration</b> to approximate uniqueness faster (train using sampled pairs).</li>
      <li>Click <b>Build Groups / Pairs</b>. Review group stats, the embedding plot, and export JSON/CSV.</li>
    </ol>
  </div>
  <div style="margin-top:10px;font-size:13px;color:rgba(255,255,255,0.95);">
    <b>Creators:</b>
    Ryan Childs (<a href="mailto:ryanchilds10@gmail.com">ryanchilds10@gmail.com</a>) ·
    James Quandt (<a href="mailto:archdukequandt@gmail.com">archdukequandt@gmail.com</a>) ·
    James Belhund (<a href="mailto:jamesbelhund@gmail.com">jamesbelhund@gmail.com</a>)
  </div>
</div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("What the discrete-geometry scoring is doing (formulas + details)", expanded=False):
        st.markdown(
            r'''
### 1) Convert each item into a multi-axis shape (radar polygon)

Pick **d** numeric features. After normalization to **[0,1]**, each item becomes a set of radii:

\[
\mathbf{v} = (v_0, v_1, \dots, v_{d-1}),\quad v_k \in [0,1]
\]

Angles are evenly spaced:

\[
\theta_k = \frac{2\pi k}{d}
\]

Polygon vertices in 2D:

\[
x_k = v_k \cos(\theta_k),\quad y_k = v_k \sin(\theta_k)
\]

This yields a *radar* / *spider* polygon for each item.

---

### 2) Discrete geometry: rasterize each polygon onto a grid

We create a uniform grid over \([-s, s]\times[-s, s]\) (resolution = **N×N**).  
For each grid cell center \((x, y)\), we mark it as inside the polygon using a point-in-polygon test.  
That produces a boolean mask \(M\) representing the polygon’s occupied area.

---

### 3) Uniqueness between two items = non-overlapping area (Jaccard complement)

Given two masks \(M_A\) and \(M_B\):

- Intersection area (overlap): \(|M_A \cap M_B|\)
- Union area: \(|M_A \cup M_B|\)

Jaccard overlap:

\[
J(A,B) = \frac{|M_A \cap M_B|}{|M_A \cup M_B|}
\]

Uniqueness score:

\[
U(A,B) = 1 - J(A,B)
\]

So **higher U** means **less overlap** (more unique).

---

### 4) Group objective: maximize uniqueness, balance groups

Within each group, we compute average pairwise uniqueness:

\[
S(G) = \frac{1}{\binom{|G|}{2}} \sum_{i<j,\ i,j\in G} U(i,j)
\]

Across groups \(G_1,\dots,G_m\), the app optimizes:

\[
\text{Objective} = \operatorname{mean}(S) - \lambda\,\operatorname{var}(S)
\]

- \(\lambda\) = “variance penalty” slider  
- This encourages **high overall uniqueness** while preventing one group from being great and others being poor.

---

### 5) Keras acceleration (optional)

Computing exact \(U(i,j)\) for many candidate swaps is expensive.  
When enabled, the app trains a small neural network to approximate uniqueness:

- Input: concatenated normalized features \([\mathbf{x}_i, \mathbf{x}_j]\)
- Target: exact \(U(i,j)\) from the discrete geometry masks  
- Then the optimizer uses the NN’s predicted uniqueness for faster iterations.
            '''
        )

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Dataset")
        data_path = st.text_input(
            "CSV path",
            value=DATA_DEFAULT,
            help="Place your CSV next to app.py or provide an absolute path.",
            key="data_path",
        )

        st.header("Modes")
        pair_only = st.checkbox(
            "Pair-only mode (perfect matching)",
            value=False,
            help="Builds disjoint pairs instead of multi-item groups.",
            key="pair_only",
        )

        st.header("Grouping")
        group_size = st.number_input(
            "Items per group",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="group_size",
        )
        n_groups = st.number_input(
            "Number of groups",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            key="n_groups",
        )
        var_weight = st.slider(
            "Balance groups (variance penalty)",
            0.0,
            2.0,
            0.25,
            0.05,
            key="var_weight",
        )
        iters = st.number_input(
            "Optimization iterations",
            min_value=200,
            max_value=20000,
            value=3000,
            step=200,
            key="iters",
        )

        st.header("Geometry")
        grid_n = st.slider(
            "Discrete geometry resolution",
            80,
            240,
            140,
            10,
            key="grid_n",
        )

        st.header("Neural net")
        use_nn = st.checkbox(
            "Use Keras model to approximate uniqueness",
            value=True,
            key="use_nn",
        )
        train_pairs = st.number_input(
            "Training pairs sampled",
            min_value=2000,
            max_value=200000,
            value=25000,
            step=1000,
            key="train_pairs",
        )
        epochs = st.number_input(
            "NN epochs",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
            key="epochs",
        )

        st.header("Team diversity model (secondary NN)")
        use_team_nn = st.checkbox(
            "Use secondary Keras model to score whole-team non-overlap",
            value=True,
            help="Trains a second neural net to predict team-level diversity (exclusive/union) from member features.",
            key="use_team_nn",
        )
        team_train_groups = st.number_input(
            "Team-model training groups sampled",
            min_value=500,
            max_value=200000,
            value=15000,
            step=500,
            key="team_train_groups",
        )
        team_epochs = st.number_input(
            "Team-model epochs",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
            key="team_epochs",
        )
        st.caption("Objective weights (higher = emphasized during optimization)")
        team_w = st.slider("Weight: team non-overlap", 0.0, 1.0, 0.6, 0.05, key="team_w")
        pair_w = st.slider("Weight: within-team pair uniqueness", 0.0, 1.0, 0.4, 0.05, key="pair_w")

        train_btn = st.button("Train / Re-train NN", type="secondary", key="train_btn")
        train_team_btn = st.button("Train / Re-train Team NN", type="secondary", key="train_team_btn")
        build_btn = st.button("Build Groups / Pairs", type="primary", key="build_btn")

    # ---------------- Data ----------------
    df = load_data(data_path)
    st.write(f"Rows: **{len(df):,}**, Columns: **{df.shape[1]}**")

    num_cols = numeric_columns(df)

    with st.expander("Select axes (features) used to create shapes", expanded=True):
        default_cols = num_cols[:8] if len(num_cols) >= 8 else num_cols
        feature_cols = st.multiselect(
            "Feature columns",
            options=num_cols,
            default=default_cols,
            help="These numeric columns become axes on the radar polygon; values are normalized to [0,1].",
            key="feature_cols",
        )

    if not feature_cols or len(feature_cols) < 2:
        st.error("Please select at least 2 numeric feature columns.")
        st.stop()

    X = prepare_matrix(df, feature_cols)
    good = X.notna().any(axis=1)
    df2 = df.loc[good].reset_index(drop=True)
    X = X.loc[good].reset_index(drop=True)

    # Fill remaining NaNs with medians; remove infs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X.values.astype(float))
    Xn = np.clip(Xn, 0.0, 1.0)

    st.success(f"Prepared matrix: {Xn.shape[0]:,} items × {Xn.shape[1]} axes")


    # ---------------- Geometry masks (cached) ----------------
    cache_path = _mask_cache_path(data_path, feature_cols, int(grid_n))
    masks = _load_masks_from_disk(cache_path, n_items=int(Xn.shape[0]), grid_n=int(grid_n))
    if masks is None:
        with st.spinner("Building discrete geometry masks (first run / cache miss)..."):
            masks = compute_masks(Xn, grid_n=int(grid_n), span=1.05)
        try:
            _save_masks_to_disk(cache_path, masks, grid_n=int(grid_n), span=1.05)
        except Exception:
            # Disk cache is best-effort; ignore failures (e.g., read-only environments).
            pass
    else:
        st.info("Loaded geometric masks from cache.")

    exact_score = make_exact_pair_scorer(masks)

    # ---------------- Neural net (optional) ----------------
    model = None
    if use_nn and nn_model.HAS_TF:
        if "pair_model" in st.session_state and not train_btn:
            model = st.session_state["pair_model"]

        if train_btn or model is None:
            with st.spinner("Sampling training pairs & computing exact uniqueness..."):
                rng = np.random.default_rng(7)
                n = Xn.shape[0]
                A = rng.integers(0, n, size=int(train_pairs))
                B = rng.integers(0, n, size=int(train_pairs))
                mask_ab = A != B
                A = A[mask_ab]
                B = B[mask_ab]
                y = np.array([exact_score(int(i), int(j)) for i, j in zip(A, B)], dtype=np.float32)

                XA = Xn[A]
                XB = Xn[B]
                Xp = np.concatenate([XA, XB], axis=1).astype(np.float32)

            with st.spinner("Training neural net..."):
                model = nn_model.build_pair_model(input_dim=Xn.shape[1])
                hist = model.fit(
                    Xp,
                    y,
                    validation_split=0.15,
                    epochs=int(epochs),
                    batch_size=512,
                    verbose=0,
                )
                st.session_state["pair_model"] = model

            st.info(
                f"Trained NN. Final loss={hist.history['loss'][-1]:.4f}, "
                f"val_loss={hist.history['val_loss'][-1]:.4f}"
            )
    else:
        if use_nn and not nn_model.HAS_TF:
            st.warning("TensorFlow/Keras not available; using exact geometry scoring only.")

    # Scorer used in optimization
    if model is not None:
        cache = {}

        def score(i: int, j: int) -> float:
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            XA = Xn[key[0]].reshape(1, -1)
            XB = Xn[key[1]].reshape(1, -1)
            Xp = np.concatenate([XA, XB], axis=1).astype(np.float32)
            pred = float(model.predict(Xp, verbose=0)[0][0])
            pred = max(0.0, min(1.0, pred))
            cache[key] = pred
            return pred

        scorer_name = "Keras predicted uniqueness"
    else:
        score = exact_score
        scorer_name = "Exact geometry uniqueness"

    st.markdown(f"### Scoring mode: **{scorer_name}**")
    st.caption(
        "Group score combines team-level non-overlap with within-team pair uniqueness. "
        "Train the secondary team model for faster/more nuanced team scoring."
    )

    # ---------------- Secondary team model (optional) ----------------
    # Effective group size (pair-only => teams of size 2)
    effective_gs = 2 if pair_only else int(group_size)

    # Precompute flat (0/1) masks once for fast team-diversity labels & scoring
    # Use cache_path (already hashed from data_path/features/grid) to key this large in-memory matrix
    flat_key = (str(cache_path), int(grid_n), int(Xn.shape[0]))
    if st.session_state.get("_flat_masks_key") != flat_key:
        with st.spinner("Preparing fast mask matrix for team scoring..."):
            flat_masks = np.stack([m.reshape(-1).astype(np.uint8) for m in masks], axis=0)
        st.session_state["_flat_masks_key"] = flat_key
        st.session_state["_flat_masks"] = flat_masks
    else:
        flat_masks = st.session_state.get("_flat_masks")

    def _team_diversity_exact_fast(indices):
        """Exact exclusive/union using fast flattened masks (no 2D grid needed)."""
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return 0.0
        cnt = flat_masks[idx].sum(axis=0)
        union = float((cnt > 0).sum())
        if union <= 0:
            return 0.0
        excl = float((cnt == 1).sum())
        return float(excl / union)

    team_model = None
    if use_team_nn and nn_model.HAS_TF:
        # Load existing model from session (if compatible)
        tm = st.session_state.get("team_model")
        tm_size = st.session_state.get("team_model_size")
        if tm is not None and int(tm_size or -1) == int(effective_gs) and not train_team_btn:
            team_model = tm

        # Train / retrain if requested or missing/incompatible
        if train_team_btn or team_model is None:
            if effective_gs < 2:
                st.warning("Team model requires team size >= 2")
            else:
                with st.spinner("Sampling training teams & computing exact team non-overlap labels..."):
                    rng = np.random.default_rng(11)
                    n_items = int(Xn.shape[0])
                    n_groups_samp = int(team_train_groups)
                    # sample groups (with replacement across groups, without replacement within each group)
                    groups_idx = np.zeros((n_groups_samp, effective_gs), dtype=int)
                    for k in range(n_groups_samp):
                        groups_idx[k] = rng.choice(n_items, size=effective_gs, replace=False)
                        rng.shuffle(groups_idx[k])

                    # Build X tensor for the team model: (G, gs, D)
                    Xg = Xn[groups_idx].astype(np.float32)

                    # Exact team diversity label in batches using flat masks
                    y_team = np.zeros((n_groups_samp,), dtype=np.float32)
                    bs = 256
                    for b0 in range(0, n_groups_samp, bs):
                        b1 = min(n_groups_samp, b0 + bs)
                        m_batch = flat_masks[groups_idx[b0:b1]]  # (B,gs,bits)
                        cnt = m_batch.sum(axis=1)  # (B,bits)
                        union = (cnt > 0).sum(axis=1).astype(np.float32)
                        excl = (cnt == 1).sum(axis=1).astype(np.float32)
                        div = np.divide(excl, np.maximum(union, 1.0))
                        y_team[b0:b1] = div

                    # Second target (pairwise uniqueness) is trained as a helpful auxiliary signal.
                    # We compute it efficiently when the pair NN exists; otherwise we approximate.
                    if model is not None:
                        # compute mean pairwise uniqueness across all sampled groups in one batched predict
                        pos_pairs = [(i, j) for i in range(effective_gs) for j in range(i + 1, effective_gs)]
                        pa = np.array([p[0] for p in pos_pairs], dtype=int)
                        pb = np.array([p[1] for p in pos_pairs], dtype=int)
                        A = groups_idx[:, pa].reshape(-1)
                        B = groups_idx[:, pb].reshape(-1)
                        XA = Xn[A]
                        XB = Xn[B]
                        Xp = np.concatenate([XA, XB], axis=1).astype(np.float32)
                        with st.spinner("Computing auxiliary pairwise targets via pair-NN (batched)..."):
                            y_pairs = model.predict(Xp, batch_size=4096, verbose=0).reshape(-1)
                        y_pair = y_pairs.reshape(n_groups_samp, -1).mean(axis=1).astype(np.float32)
                    else:
                        # fallback: use normalized dispersion proxy from features
                        y_pair = np.mean(np.abs(Xg - Xg.mean(axis=1, keepdims=True)), axis=(1, 2)).astype(np.float32)
                        y_pair = np.clip(y_pair, 0.0, 1.0)

                with st.spinner("Training secondary team NN..."):
                    team_model = nn_model.build_team_model(input_dim=int(Xn.shape[1]), team_size=int(effective_gs))
                    hist = team_model.fit(
                        Xg,
                        {"team_diversity": y_team, "mean_pair_uniqueness": y_pair},
                        validation_split=0.15,
                        epochs=int(team_epochs),
                        batch_size=256,
                        verbose=0,
                    )
                    st.session_state["team_model"] = team_model
                    st.session_state["team_model_size"] = int(effective_gs)

                st.info(
                    "Trained team NN. "
                    f"Final loss={hist.history['loss'][-1]:.4f}, "
                    f"val_loss={hist.history['val_loss'][-1]:.4f}"
                )
    else:
        if use_team_nn and not nn_model.HAS_TF:
            st.warning("TensorFlow/Keras not available; team-level scoring will use exact geometry instead of the secondary NN.")

    # Build a group-level scorer that jointly rewards team non-overlap and pairwise uniqueness
    # (this is what makes groups 'equally diverse' in the team sense, not just pairwise.)
    w_sum = float(team_w) + float(pair_w)
    tw = float(team_w) / w_sum if w_sum > 0 else 0.6
    pw = float(pair_w) / w_sum if w_sum > 0 else 0.4

    team_scorer_name = "Secondary team NN" if team_model is not None else "Exact team geometry"
    st.markdown(
        f"**Group score:** {tw:.2f}×team non-overlap + {pw:.2f}×pair uniqueness · "
        f"**Team scorer:** {team_scorer_name}"
    )

    _group_cache = {}

    def group_score_fn(group_indices):
        key = tuple(sorted([int(x) for x in group_indices]))
        if key in _group_cache:
            return _group_cache[key]
        # pairwise uniqueness component (uses chosen scorer: exact or pair-NN)
        pairs = list(combinations(key, 2))
        pair_mean = float(np.mean([score(i, j) for i, j in pairs])) if pairs else 0.0
        # team non-overlap component
        if team_model is not None:
            Xg1 = Xn[np.array(key, dtype=int)].reshape(1, effective_gs, -1).astype(np.float32)
            t_pred, p_pred = team_model.predict(Xg1, verbose=0)
            team_div = float(t_pred.reshape(-1)[0])
        else:
            team_div = float(_team_diversity_exact_fast(key))
        combined = tw * team_div + pw * pair_mean
        _group_cache[key] = float(combined)
        return float(combined)

    # ---------------- Results persistence (prevents re-running when selecting teams) ----------------
    def _results_fingerprint() -> str:
        payload = {
            "data_path": str(data_path),
            "n_rows": int(df2.shape[0]),
            "feature_cols": list(feature_cols),
            "grid_n": int(grid_n),
            "pair_only": bool(pair_only),
            "group_size": int(group_size),
            "n_groups": int(n_groups),
            "var_weight": float(var_weight),
            "iters": int(iters),
            "use_nn": bool(use_nn and nn_model.HAS_TF),
            "train_pairs": int(train_pairs),
            "epochs": int(epochs),
            "use_team_nn": bool(use_team_nn and nn_model.HAS_TF),
            "team_train_groups": int(team_train_groups),
            "team_epochs": int(team_epochs),
            "team_w": float(team_w),
            "pair_w": float(pair_w),
        }
        s = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    fp = _results_fingerprint()

    if build_btn:
        if pair_only:
            gs = 2
            ng = Xn.shape[0] // 2
        else:
            gs = int(group_size)
            ng = int(n_groups)

        with st.spinner("Optimizing groups..."):
            groups0 = random_groups(Xn.shape[0], gs, ng)
            best_groups, best_obj, meta2 = hillclimb(
                groups0,
                pair_score_fn=score,
                group_score_fn=group_score_fn,
                n_items=Xn.shape[0],
                iters=int(iters),
                var_weight=float(var_weight),
            )

        st.session_state["last_results"] = {
            "fingerprint": fp,
            "best_groups": best_groups,
            "best_obj": float(best_obj),
            "meta2": meta2,
            "gs": int(gs),
            "ng": int(ng),
        }

    results = st.session_state.get("last_results")
    if not results or results.get("fingerprint") != fp:
        st.info("Click **Build Groups / Pairs** to generate groups. (Results are cached for selection & visualization.)")
        return

    best_groups = results["best_groups"]
    best_obj = float(results["best_obj"])
    meta2 = results["meta2"]
    gs = int(results["gs"])
    ng = int(results["ng"])

    st.success(
        f"Optimization complete. Objective={best_obj:.4f}  "
        f"(mean={meta2['mean']:.4f}, var={meta2['var']:.6f})"
    )

    # Group summary (exact scoring for reporting)
    rows = []
    for gi, g in enumerate(best_groups):
        g_pairs = list(combinations(g, 2))
        ex = float(np.mean([exact_score(i, j) for i, j in g_pairs])) if g_pairs else 0.0
        team_div_ex = float(_team_diversity_exact_fast(g))
        combined_ex = tw * team_div_ex + pw * ex
        rows.append(
            {
                "group": gi + 1,
                "members": len(g),
                "mean_pairwise_uniqueness_exact": ex,
                "team_diversity_exact": team_div_ex,
                "combined_score_exact": combined_ex,
            }
        )

    summ = pd.DataFrame(rows).sort_values("combined_score_exact", ascending=False)
    st.dataframe(summ, use_container_width=True)

    # Embedding visualization (PCA)
    st.markdown("### Embedding Visualization")
    try:
        from sklearn.decomposition import PCA

        Z = PCA(n_components=2).fit_transform(Xn)
        viz_df = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1]})
        viz_df["group"] = -1
        for gi, g in enumerate(best_groups):
            for idx2 in g:
                viz_df.loc[idx2, "group"] = gi + 1
        st.scatter_chart(viz_df, x="x", y="y", color="group")
    except Exception as e:
        st.warning(f"Embedding visualization unavailable: {e}")

    # ---------------- Team inspection ----------------
    st.markdown("### Team Inspector")
    pick = st.selectbox(
        "Inspect team",
        options=list(range(1, len(best_groups) + 1)),
        index=int(st.session_state.get("inspect_group", 1)) - 1 if len(best_groups) else 0,
        key="inspect_group",
    )
    g = best_groups[pick - 1]
    detail = df2.iloc[g].copy()
    detail["_idx"] = g
    st.markdown("#### Team members")
    cols_to_show = ["Name"] if "Name" in detail.columns else []
    cols_to_show += feature_cols + ["_idx"]
    cols_to_show = [c for c in cols_to_show if c in detail.columns]
    st.dataframe(detail[cols_to_show], use_container_width=True)

    # Optionally open multiple teams without rebuilding (useful for side-by-side checks)
    with st.expander("Open multiple teams (side-by-side)", expanded=False):
        open_teams = st.multiselect(
            "Teams to open",
            options=list(range(1, len(best_groups) + 1)),
            default=[pick],
            help="Selecting teams here will not re-run optimization; it only changes what is displayed.",
            key="open_teams",
        )
        grid_n_int_local = int(grid_n)
        preview_rows = []
        for t in open_teams:
            gg = best_groups[t - 1]
            gg_pairs = list(combinations(gg, 2))
            mean_u = float(np.mean([exact_score(i, j) for i, j in gg_pairs])) if gg_pairs else 0.0

            # quick discrete geometry stats
            cnt = np.zeros((grid_n_int_local, grid_n_int_local), dtype=np.int16)
            for ii in gg:
                cnt += masks[ii].astype(np.int16)
            u_area = int((cnt > 0).sum())
            excl_area = int((cnt == 1).sum())
            div = (excl_area / u_area) if u_area > 0 else 0.0
            preview_rows.append(
                {
                    "team": int(t),
                    "members": int(len(gg)),
                    "mean_pairwise_uniqueness_exact": mean_u,
                    "union_cells": u_area,
                    "exclusive_cells": excl_area,
                    "exclusive/union": div,
                }
            )
        st.dataframe(pd.DataFrame(preview_rows).sort_values("mean_pairwise_uniqueness_exact", ascending=False), use_container_width=True)

    with st.expander("Verify items (open in browser)", expanded=False):
        names = detail["Name"].astype(str).tolist() if "Name" in detail.columns else [str(i) for i in g]
        for name in names:
            q = name.replace(" ", "+")
            st.markdown(f"- [{name}](https://www.google.com/search?q={q})")

    # ---------------- 3D discrete-geometry visualization ----------------
    st.markdown("### 3D Team Geometry Viewer (Discrete Masks)")
    st.caption("Shows each member’s rasterized polygon as a stacked 3D point cloud, plus an overlap surface.")
    max_points = st.slider("Max points per member (downsample for speed)", 500, 20000, 5000, 500, key="max_points_3d")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _mask_points(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, max_pts: int, seed: int = 0):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.array([]), np.array([])
        if coords.shape[0] > max_pts:
            rng = np.random.default_rng(seed)
            sel = rng.choice(coords.shape[0], size=int(max_pts), replace=False)
            coords = coords[sel]
        # coords are (iy, ix)
        x = xs[coords[:, 1]]
        y = ys[coords[:, 0]]
        return x, y

    grid_n_int = int(grid_n)
    span = 1.05
    xs = np.linspace(-span, span, grid_n_int)
    ys = np.linspace(-span, span, grid_n_int)

    # Compute team overlap grid
    count = np.zeros((grid_n_int, grid_n_int), dtype=np.int16)
    for idx2 in g:
        count += masks[idx2].astype(np.int16)

    union_area = int((count > 0).sum())
    overlap_area = int((count > 1).sum())
    exclusive_area = int((count == 1).sum())
    team_diversity = (exclusive_area / union_area) if union_area > 0 else 0.0

    # Team pairwise uniqueness
    team_pairs = list(combinations(g, 2))
    team_pairwise_exact = float(np.mean([exact_score(i, j) for i, j in team_pairs])) if team_pairs else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Union area (cells)", f"{union_area:,}")
    c2.metric("Exclusive area (cells)", f"{exclusive_area:,}")
    c3.metric("Overlap area (cells)", f"{overlap_area:,}")
    c4.metric("Team diversity (exclusive/union)", f"{team_diversity:.3f}")

    st.metric("Mean pairwise uniqueness (exact)", f"{team_pairwise_exact:.4f}")

    tab_layers, tab_overlap = st.tabs(["Member layers (3D)", "Overlap surface (3D)"])

    with tab_layers:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("member layer")

        for layer, idx2 in enumerate(g):
            x, y = _mask_points(masks[idx2], xs, ys, int(max_points), seed=int(idx2))
            if x.size == 0:
                continue
            z = np.full_like(x, layer, dtype=float)
            ax.scatter(x, y, z, s=2, alpha=0.65)

        ax.set_title(f"Team {pick}: stacked member masks (downsampled)")
        st.pyplot(fig, clear_figure=True)

    with tab_overlap:
        # Overlap count surface
        Xg, Yg = np.meshgrid(xs, ys)
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("overlap count")
        # Downsample grid for speed
        stride = max(1, grid_n_int // 80)
        ax2.plot_surface(
            Xg[::stride, ::stride],
            Yg[::stride, ::stride],
            count[::stride, ::stride].astype(float),
            linewidth=0,
            antialiased=False,
            alpha=0.9,
        )
        ax2.set_title(f"Team {pick}: overlap count surface")
        st.pyplot(fig2, clear_figure=True)

    # Per-member discrete geometry stats
    with st.expander("Per-member discrete geometry stats", expanded=False):
        member_rows = []
        for idx2 in g:
            m = masks[idx2]
            area = int(m.sum())
            excl = int(np.logical_and(m, count == 1).sum())
            member_rows.append(
                {
                    "idx": int(idx2),
                    "name": str(df2.loc[idx2, "Name"]) if "Name" in df2.columns else str(idx2),
                    "area_cells": area,
                    "exclusive_cells": excl,
                    "exclusive_share": (excl / area) if area > 0 else 0.0,
                }
            )
        st.dataframe(pd.DataFrame(member_rows).sort_values("exclusive_cells", ascending=False), use_container_width=True)

    # Export JSON
    export = {
        "feature_cols": feature_cols,
        "pair_only": bool(pair_only),
        "group_size": int(gs),
        "n_groups": int(ng),
        "var_weight": float(var_weight),
        "team_weight": float(tw),
        "pair_weight": float(pw),
        "objective": float(best_obj),
        "group_scores": meta2["group_scores"],
        "group_metrics_exact": rows,
        "groups": [[int(i) for i in gg] for gg in best_groups],
    }
    st.download_button(
        "Download groups JSON",
        data=json.dumps(export, indent=2),
        file_name="groups.json",
        mime="application/json",
    )

    # Export CSV
    csv_rows = []
    for gi, gg in enumerate(best_groups):
        for idx2 in gg:
            row = df2.iloc[idx2].to_dict()
            row["group"] = gi + 1
            csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)
    st.download_button(
        "Download groups CSV",
        data=csv_df.to_csv(index=False),
        file_name="groups.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
