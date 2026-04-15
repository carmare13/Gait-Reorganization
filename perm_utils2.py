import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def _distance_covariance_squared(x, y):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    n = x.size
    if n < 2:
        return 0.0

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2 = np.mean(A * B)
    return max(float(dcov2), 0.0)


def distance_correlation_1d(x, y):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    if x.size < 2:
        return 0.0

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return 0.0

    if np.allclose(np.nanstd(x), 0) or np.allclose(np.nanstd(y), 0):
        return 0.0

    dcov2_xy = _distance_covariance_squared(x, y)
    dcov2_xx = _distance_covariance_squared(x, x)
    dcov2_yy = _distance_covariance_squared(y, y)

    if dcov2_xx <= 0 or dcov2_yy <= 0:
        return 0.0

    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy))
    return float(np.clip(dcor, 0.0, 1.0))


def calcular_dcor_single_variable(z_feature, X_var_time):
    """
    Calcula el perfil temporal de dcor para una sola variable.
    Parameters
    ----------
    z_feature : array, shape (n_samples,)
    X_var_time : array, shape (n_samples, n_time)

    Returns
    -------
    dcor_profile : array, shape (n_time,)
    """
    z_feature = np.asarray(z_feature, dtype=np.float32).ravel()
    X_var_time = np.asarray(X_var_time, dtype=np.float32)

    if X_var_time.ndim != 2:
        raise ValueError("X_var_time debe tener shape (n_samples, n_time).")
    if X_var_time.shape[0] != z_feature.shape[0]:
        raise ValueError("z_feature y X_var_time deben tener el mismo número de muestras.")

    n_time = X_var_time.shape[1]
    dcor_profile = np.zeros(n_time, dtype=np.float32)

    for t in range(n_time):
        dcor_profile[t] = distance_correlation_1d(z_feature, X_var_time[:, t])

    return dcor_profile


def calcular_distance_correlation_paralelo(z_feature, X_raw_filtrado, n_jobs=2):
    """
    Paraleliza el cálculo por variable.
    Parameters
    ----------
    z_feature : array, shape (n_samples,)
    X_raw_filtrado : array, shape (n_samples, n_time, n_features)

    Returns
    -------
    matriz_dcor : array, shape (n_time, n_features)
    """
    X_raw_filtrado = np.asarray(X_raw_filtrado, dtype=np.float32)

    if X_raw_filtrado.ndim != 3:
        raise ValueError("X_raw_filtrado debe tener shape (n_samples, n_time, n_features).")

    n_samples, n_time, n_features = X_raw_filtrado.shape

    resultados = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        verbose=0
    )(
        delayed(calcular_dcor_single_variable)(
            z_feature,
            X_raw_filtrado[:, :, i]
        )
        for i in range(n_features)
    )

    matriz_dcor = np.array(resultados, dtype=np.float32).T
    return matriz_dcor


def calcular_lri_variantes(matriz_dcor, tau=0.20, dx=None):
    """
    Calcula varias versiones del LRI.
    """
    matriz_dcor = np.asarray(matriz_dcor, dtype=np.float32)

    if dx is None:
        dx = 100.0 / (matriz_dcor.shape[0] - 1)

    lri_mean = np.mean(matriz_dcor, axis=0)
    lri_auc = np.trapezoid(matriz_dcor, dx=dx, axis=0)

    matriz_thr = np.where(matriz_dcor >= tau, matriz_dcor, 0.0)
    lri_auc_thr = np.trapezoid(matriz_thr, dx=dx, axis=0)

    return {
        "lri_mean": lri_mean,
        "lri_auc": lri_auc,
        "lri_auc_thr": lri_auc_thr
    }


def una_perm_lri_3g(
    seed,
    Z_train,
    Y_train,
    Z_test,
    Y_test,
    X_test_filtrado,
    tau=0.20,
    n_jobs_dcor=2,
):
    """
    Una sola permutación.
    """
    rng = np.random.default_rng(seed)

    Z_train = np.asarray(Z_train, dtype=np.float32)
    Y_train = np.asarray(Y_train).ravel()
    Z_test = np.asarray(Z_test, dtype=np.float32)
    Y_test = np.asarray(Y_test).ravel()
    X_test_filtrado = np.asarray(X_test_filtrado, dtype=np.float32)

    if Z_train.ndim != 2:
        raise ValueError("Z_train debe tener shape (n_samples, n_latent).")
    if Z_test.ndim != 2:
        raise ValueError("Z_test debe tener shape (n_samples, n_latent).")
    if X_test_filtrado.ndim != 3:
        raise ValueError("X_test_filtrado debe tener shape (n_samples, n_time, n_features).")

    if len(Y_train) != Z_train.shape[0]:
        raise ValueError("Y_train no coincide con Z_train.")
    if len(Y_test) != Z_test.shape[0]:
        raise ValueError("Y_test no coincide con Z_test.")
    if X_test_filtrado.shape[0] != Z_test.shape[0]:
        raise ValueError("X_test_filtrado y Z_test deben tener el mismo número de muestras.")

    Y_perm = rng.permutation(Y_train)

    lda_fake = LDA(n_components=2)
    lda_fake.fit(Z_train, Y_perm)

    Z_test_lda_fake = lda_fake.transform(Z_test)
    S_age_fake = Z_test_lda_fake[:, 0].astype(np.float32, copy=False)

    matriz_dcor_fake = calcular_distance_correlation_paralelo(
        z_feature=S_age_fake,
        X_raw_filtrado=X_test_filtrado,
        n_jobs=n_jobs_dcor
    )

    lri_dict_fake = calcular_lri_variantes(
        matriz_dcor=matriz_dcor_fake,
        tau=tau
    )

    return lri_dict_fake["lri_auc_thr"]