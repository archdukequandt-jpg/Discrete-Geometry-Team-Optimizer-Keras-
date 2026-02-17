import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    HAS_TF = True
except Exception:
    HAS_TF = False

def build_pair_model(input_dim: int):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")
    inp = layers.Input(shape=(input_dim*2,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)  # predict uniqueness in [0,1]
    m = models.Model(inp, out)
    m.compile(optimizer=optimizers.Adam(learning_rate=2e-3), loss="mse")
    return m

def make_pair_X(X, idx_a, idx_b):
    a = X[idx_a]
    b = X[idx_b]
    return np.concatenate([a, b], axis=1)

def predict_pairs(model, X, pairs, batch_size=4096):
    A = np.array([p[0] for p in pairs], dtype=int)
    B = np.array([p[1] for p in pairs], dtype=int)
    XA = X[A]; XB = X[B]
    Xp = np.concatenate([XA, XB], axis=1).astype(np.float32)
    y = model.predict(Xp, batch_size=batch_size, verbose=0).reshape(-1)
    return y


def build_team_model(input_dim: int, team_size: int):
    """Build a secondary NN that scores whole teams.

    The model is designed to be fairly order-robust (not perfectly invariant,
    but strongly permutation-resistant) by combining:
      - per-member encodings aggregated with mean/max pooling
      - symmetric "set" statistics (mean, variance)
      - a light pairwise-difference summary (mean absolute diff)

    Outputs (both in [0,1]):
      - team_diversity: exclusive/union for the team mask
      - mean_pair_uniqueness: average pairwise uniqueness within the team
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")
    if team_size < 2:
        raise ValueError("team_size must be >= 2")

    inp = layers.Input(shape=(team_size, input_dim), name="team_features")  # (B,G,D)

    # Per-member encoder (shared)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(inp)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)

    # Pooling over set dimension
    mean_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="mean_pool")(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=1), name="max_pool")(x)

    # Symmetric stats directly from raw normalized features
    raw_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="raw_mean")(inp)
    raw_var = layers.Lambda(lambda t: tf.math.reduce_variance(t, axis=1), name="raw_var")(inp)

    # Lightweight pairwise dispersion summary: mean absolute difference across all pairs
    # Shapes: (B,G,1,D) - (B,1,G,D) => (B,G,G,D)
    diffs = layers.Lambda(
        lambda t: tf.abs(tf.expand_dims(t, 2) - tf.expand_dims(t, 1)),
        name="pair_abs_diffs",
    )(inp)
    pair_disp = layers.Lambda(lambda t: tf.reduce_mean(t, axis=[1, 2]), name="pair_disp")(diffs)  # (B,D)

    h = layers.Concatenate()([mean_pool, max_pool, raw_mean, raw_var, pair_disp])
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Dense(128, activation="relu")(h)

    out_team = layers.Dense(1, activation="sigmoid", name="team_diversity")(h)
    out_pair = layers.Dense(1, activation="sigmoid", name="mean_pair_uniqueness")(h)

    m = models.Model(inp, [out_team, out_pair], name="team_model")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=2e-3),
        loss={"team_diversity": "mse", "mean_pair_uniqueness": "mse"},
        loss_weights={"team_diversity": 1.0, "mean_pair_uniqueness": 0.6},
    )
    return m
