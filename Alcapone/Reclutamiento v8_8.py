"""
# Estimación del tamaño del crimen organizado mediante dinámica de sistemas
# Autores: Amador, Pérez Ricart, Favennec & Carbonell (2026)
# Versionado: modelo por estado, 2021-2025
#
# Instalación:
#   pip install numpy scipy pandas openpyxl matplotlib tqdm
#
# Ejecución:
#   python reclutamiento v8_3.py

"""


# ============================================================
# SECCIÓN 1 — CONTROL Y PARÁMETROS GLOBALES
# ============================================================

ANIO_INICIO  = 2021
ANIO_FIN     = 2025
N_ESCENARIOS = 1000000
RUTA_EXCEL   = r""  
ESTADOS_A_CORRER = []  

MESES = ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO",
         "AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"]

NAVY    = "#285275"
GREEN   = "#12B76A"
GREY_LT = "#E5E7EB"

RELIABILITY_W = dict(
    hom=3.5,   
    arm=2.5,   
    ext=1.3,
    nar=1.1,
    sec=0.9,
    sal=0.9,
    fin=0.8,
    hid=0.7,
    tra=0.7
)

# Normalización para que el promedio de pesos sea 1
_norm = len(RELIABILITY_W) / sum(RELIABILITY_W.values())
W_LOSS = {k: float(v * _norm) for k, v in RELIABILITY_W.items()}

VARS_DELITO = ["homicidios","secuestro","extorsion","salud","narcomenudeo",
               "financieros","hidrocarburos","narcotrafico","armas"]

NORM_DICT = {
    "CIUDAD DE MEXICO" : "CIUDAD DE MEXICO",
    "CDMX"             : "CIUDAD DE MEXICO",
    "COAHUILA"         : "COAHUILA DE ZARAGOZA",
    "MEXICO"           : "ESTADO DE MEXICO",
    "MICHOACAN"        : "MICHOACAN DE OCAMPO",
    "NUEVO LEON"       : "NUEVO LEON",
    "QUERETARO"        : "QUERETARO",
    "SAN LUIS POTOSI"  : "SAN LUIS POTOSI",
    "VERACRUZ"         : "VERACRUZ DE IGNACIO DE LA LLAVE",
    "YUCATAN"          : "YUCATAN",
}

# ============================================================
# SECCIÓN 2 — FUNCIONES PURAS
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def normalizar_estado(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().upper()
    for a, b in [("A","A"),("E","E"),("I","I"),("O","O"),("U","U"),("U","U"),("N","N")]:
        pass  
    s = (s.replace("\u00c1","A").replace("\u00c9","E").replace("\u00cd","I")
          .replace("\u00d3","O").replace("\u00da","U").replace("\u00dc","U")
          .replace("\u00d1","N"))
    return NORM_DICT.get(s, s)


def make_inputs_estado(n_meses):
    t    = np.arange(1, n_meses + 1, dtype=float)
    phi1, phi2 = np.pi/6, -np.pi/8
    patron     = np.array([0,0.03,0,-0.02,0,0.02,0,-0.03,0,0.02,0,-0.01])
    patron_rep = np.tile(patron, n_meses // 12 + 1)[:n_meses]
    pers = np.clip(
        0.45 + 0.15*np.sin(2*np.pi*(t/12)+phi1) + 0.05*np.sin(4*np.pi*(t/12)+phi2) + patron_rep,
        0.0, 1.0)
    dem = np.clip(
        1.00 + 0.25*np.sin(2*np.pi*(t/12)-np.pi/4) + 0.05*np.cos(2*np.pi*(t/12))
        + np.linspace(-0.05, 0.05, n_meses),
        0.6, 1.5)
    return {"n_meses": n_meses, "pers": pers, "dem": dem}


def ode_core(t, state, p, inputs):
    A, K = state
    A = max(A, 1e-9);  K = max(K, 0.0)
    idx  = int(np.clip(round(t) - 1, 0, inputs["n_meses"] - 1))
    pers = float(np.clip(inputs["pers"][idx], 0.0, 1.0))
    dem  = float(np.clip(inputs["dem"][idx],  0.6, 3.0))
    pago = K / A
    if not np.isfinite(pago): pago = 0.0
    mult_fin = float(np.clip(0.7 + 0.6*(pago/(p["salario_minimo_deseado"]+1e-9)), 0.5, 1.6))

    tasa_rec = max(p["tasa_base_reclutamiento"]
                   + p["sensibilidad_reclutamiento_pago"] * pago
                   - p["sensibilidad_reclutamiento_persecucion"] * pers
                   + p["componente_cooptacion_reclutamiento"], 0.0)
    flujo_rec = tasa_rec * A
    flujo_inc = p["tasa_incapacitacion_por_persecucion"] * pers * A
    flujo_let = p["tasa_letalidad_por_persecucion"]      * pers * A
    pago_ins  = max(p["salario_minimo_deseado"] - pago, 0.0)
    flujo_des = max((p["tasa_desercion_base"]
                     + p["sensibilidad_desercion_pago_bajo"]   * pago_ins
                     + p["sensibilidad_desercion_persecucion"] * pers) * A, 0.0)

    dA = flujo_rec - flujo_inc - flujo_let - flujo_des
    if not np.isfinite(dA): dA = -0.1 * A
    if A + dA < 0:          dA = -A

    dK = (p["productividad_ingresos_por_miembro"] * dem * A
          - (p["salario_minimo_deseado"] + p["costos_operativos_unitarios"]) * A
          - p["proporcion_decomiso_caja"] * pers * K)
    if not np.isfinite(dK): dK = -0.1 * K
    if K + dK < 0:          dK = -K
    return [dA, dK]


def simulate_estado(A0, K0, p, inputs):
    from scipy.integrate import solve_ivp
    n = inputs["n_meses"]
    try:
        sol = solve_ivp(
            fun      = lambda t, y: ode_core(t, y, p, inputs),
            t_span   = (1.0, float(n)),
            y0       = [max(A0, 1e3), max(K0, 0.0)],
            t_eval   = np.arange(1, n+1, dtype=float),
            method   = "LSODA", rtol=1e-4, atol=1e-6, max_step=1.0)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return None
        return sol
    except Exception:
        return None


def calc_observables(sol, p, inputs):
    A    = np.maximum(sol.y[0], 1e-9)
    K    = np.maximum(sol.y[1], 0.0)
    idx  = np.clip(np.round(sol.t).astype(int) - 1, 0, inputs["n_meses"]-1)
    pers = np.clip(inputs["pers"][idx], 0.0, 1.0)
    dem  = np.clip(inputs["dem"][idx],  0.6, 3.0)
    pago = K / A
    mf   = np.clip(0.7 + 0.6*(pago/(p["salario_minimo_deseado"]+1e-9)), 0.5, 1.6)
    comp = p["competencia_interorganizacional"]
    return {
        "homicidios"   : np.clip(p["Tasa_reporte_homicidio"]            * p["productividad_homicidios"]             * A*(pers+comp), 0, None),
        "secuestro"    : np.clip(p["Tasa_reporte_secuestro"]            * p["tasa_secuestros_por_miembro"]           * A*dem,         0, None),
        "extorsion"    : np.clip(p["Tasa_reporte_extorsion"]            * p["tasa_extorsiones_por_miembro"]          * A*dem,         0, None),
        "salud"        : np.clip(p["Tasa_reporte_salud"]                * p["tasa_salud_por_miembro"]                * A*dem,         0, None),
        "narcomenudeo" : np.clip(p["Tasa_reporte_narcomenudeo"]         * p["tasa_narcomenudeo_por_miembro"]         * A*dem,         0, None),
        "armas"        : np.clip(p["Tasa_reporte_armas_explosivos"]     * p["tasa_armas_explosivos_por_miembro"]     * A*(pers+comp), 0, None),
        "financieros"  : np.clip(p["Tasa_reporte_financieros_fiscales"] * p["tasa_financieros_fiscales_por_miembro"] * A*mf,          0, None),
        "hidrocarburos": np.clip(p["Tasa_reporte_hidrocarburos"]        * p["tasa_hidrocarburos_por_miembro"]        * A*(dem**0.7),  0, None),
        "trafico"      : np.clip(p["Tasa_reporte_narcotrafico"]         * p["tasa_narcotrafico_por_miembro"]         * A*dem,         0, None),
    }



# ===========================
# Pérdidas (loss) — versión para conteos mensuales
# ===========================
# LOSS_MODE:
#   - "rel_smooth": error relativo con denominador suavizado (robusto a conteos pequeños)
#   - "poisson":    -loglik Poisson (conteos)
#   - "nb":         -loglik NegBin (sobredispersión), k estimado por MOM
LOSS_MODE = "nb" # "nb", "poisson", "rel_smooth"

# Penalización suave para predicciones por debajo de un piso cuando hay eventos observados
FLOOR_BY_VAR = {
    "homicidios": 1.0, "secuestro": 1.0, "extorsion": 1.0, "salud": 1.0,
    "narcomenudeo": 1.0, "armas": 1.0, "financieros": 1.0,
    "hidrocarburos": 1.0, "narcotrafico": 1.0,
}

def compute_denom_floor(obs_real_dict, q=0.25, lo=1.0, hi=25.0):
    """Piso de denominador para error relativo.
    Se calcula por variable (y por estado) a partir de los observados mensuales > 0.
    """
    denom = {}
    for v in VARS_DELITO:
        x = np.asarray(obs_real_dict.get(v, []), dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size == 0:
            denom[v] = float(lo)
        else:
            c = float(np.quantile(x, q))
            denom[v] = float(np.clip(c, lo, hi))
    return denom

def compute_nb_k(obs_real_dict, min_k=0.5, max_k=200.0):
    """Estimación MOM de la dispersión (k) para NegBin por variable y estado.
    Si var <= mean, la serie se comporta ~Poisson => k grande (aprox. Poisson).
    """
    k = {}
    for v in VARS_DELITO:
        y = np.asarray(obs_real_dict.get(v, []), dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            k[v] = float(max_k)
            continue
        mu = float(np.mean(y))
        var = float(np.var(y))
        if mu <= 1e-9:
            k[v] = float(max_k)
        elif var > mu + 1e-9:
            kk = (mu * mu) / (var - mu)
            k[v] = float(np.clip(kk, min_k, max_k))
        else:
            k[v] = float(max_k)
    return k

def loss_rel_smooth(obs_pred, obs_real, denom_floor):
    pairs = [("hom","homicidios"),("sec","secuestro"),("ext","extorsion"),
             ("sal","salud"),("nar","narcomenudeo"),("arm","armas"),
             ("fin","financieros"),("hid","hidrocarburos"),("tra","narcotrafico")]
    total = 0.0
    for wk, vk in pairs:
        p = np.asarray(obs_pred[vk], dtype=float)
        o = np.asarray(obs_real[vk], dtype=float)

        mask = np.isfinite(o) & (o > 0) & np.isfinite(p)
        if not mask.any():
            continue

        c = float(denom_floor.get(vk, 1.0))
        denom = np.maximum(o[mask], c)
        rel_sq = ((p[mask] - o[mask]) / denom) ** 2
        fit_term = float(np.mean(rel_sq))  

        floor = float(FLOOR_BY_VAR.get(vk, 1.0))
        pen = np.where(p[mask] < floor, 10.0 * ((floor - p[mask]) / floor) ** 2, 0.0)
        pen_term = float(np.mean(pen))

        total += float(W_LOSS[wk]) * (fit_term + pen_term)

    return float(total) if np.isfinite(total) else 1e12

def loss_poisson(obs_pred, obs_real):
    """NLL Poisson (constantes omitidas)."""
    eps = 1e-9
    pairs = [("hom","homicidios"),("sec","secuestro"),("ext","extorsion"),
             ("sal","salud"),("nar","narcomenudeo"),("arm","armas"),
             ("fin","financieros"),("hid","hidrocarburos"),("tra","narcotrafico")]
    total = 0.0
    for wk, vk in pairs:
        mu = np.asarray(obs_pred[vk], dtype=float)
        y  = np.asarray(obs_real[vk], dtype=float)

        mask = np.isfinite(y) & (y >= 0) & np.isfinite(mu)
        if not mask.any():
            continue

        mu_m = np.maximum(mu[mask], eps)
        y_m  = y[mask]
        nll = np.mean(mu_m - y_m * np.log(mu_m))
        total += float(W_LOSS[wk]) * float(nll)

    return float(total) if np.isfinite(total) else 1e12

def loss_nb(obs_pred, obs_real, nb_k):
    """NLL NegBin (NB2, var = mu + mu^2/k). Incluye constantes (no afecta óptimo).
    k se estima por MOM por estado/variable.
    """
    from scipy.special import gammaln  
    eps = 1e-9
    pairs = [("hom","homicidios"),("sec","secuestro"),("ext","extorsion"),
             ("sal","salud"),("nar","narcomenudeo"),("arm","armas"),
             ("fin","financieros"),("hid","hidrocarburos"),("tra","narcotrafico")]
    total = 0.0
    for wk, vk in pairs:
        mu = np.asarray(obs_pred[vk], dtype=float)
        y  = np.asarray(obs_real[vk], dtype=float)

        mask = np.isfinite(y) & (y >= 0) & np.isfinite(mu)
        if not mask.any():
            continue

        mu_m = np.maximum(mu[mask], eps)
        y_m  = y[mask]
        k    = float(nb_k.get(vk, 50.0))
        k    = max(k, 0.5)

        # log PMF NB: Γ(y+k)-Γ(k)-Γ(y+1) + k log(k/(k+mu)) + y log(mu/(k+mu))
        ll = (
            gammaln(y_m + k) - gammaln(k) - gammaln(y_m + 1.0)
            + k * (np.log(k) - np.log(k + mu_m))
            + y_m * (np.log(mu_m) - np.log(k + mu_m))
        )
        nll = float(np.mean(-ll))
        total += float(W_LOSS[wk]) * nll

    return float(total) if np.isfinite(total) else 1e12

def total_loss(obs_pred, obs_real, denom_floor=None, nb_k=None, mode=None):
    """Selector de pérdida."""
    m = (mode or LOSS_MODE).lower()
    if m == "rel_smooth":
        if denom_floor is None:
            denom_floor = compute_denom_floor(obs_real)
        return loss_rel_smooth(obs_pred, obs_real, denom_floor)
    if m == "poisson":
        return loss_poisson(obs_pred, obs_real)
    if m == "nb":
        if nb_k is None:
            nb_k = compute_nb_k(obs_real)
        return loss_nb(obs_pred, obs_real, nb_k)
    raise ValueError(f"LOSS_MODE desconocido: {mode or LOSS_MODE}")


def draw_params(seed):
    r = np.random.default_rng(seed=seed)
    return {
        "tasa_base_reclutamiento"               : r.uniform(0.020, 0.035),
        "sensibilidad_reclutamiento_pago"       : r.uniform(0.001, 0.010),
        "sensibilidad_reclutamiento_persecucion": r.uniform(0.006, 0.015),
        "componente_cooptacion_reclutamiento"   : r.uniform(0.000, 0.003),
        "tasa_incapacitacion_por_persecucion"   : r.uniform(0.005, 0.010),
        "tasa_letalidad_por_persecucion"        : r.uniform(0.0006, 0.0015),
        "tasa_desercion_base"                   : r.uniform(0.0015, 0.005),
        "sensibilidad_desercion_pago_bajo"      : r.uniform(0.010, 0.035),
        "sensibilidad_desercion_persecucion"    : r.uniform(0.004, 0.010),
        "productividad_ingresos_por_miembro"    : r.uniform(0.20,  0.60),
        "salario_minimo_deseado"                : r.uniform(0.30,  0.60),
        "costos_operativos_unitarios"           : r.uniform(0.15,  0.35),
        "proporcion_decomiso_caja"              : r.uniform(0.03,  0.10),
        "productividad_homicidios"              : r.uniform(0.009, 0.050),
        "tasa_extorsiones_por_miembro"          : r.uniform(0.003, 0.018),
        "tasa_secuestros_por_miembro"           : r.uniform(0.00006, 0.00035),
        "competencia_interorganizacional"       : r.uniform(0.08,  0.16),
        "Tasa_reporte_extorsion"                : r.uniform(0.30,  0.45),
        "Tasa_reporte_secuestro"                : r.uniform(0.75,  0.90),
        "Tasa_reporte_homicidio"                : r.uniform(0.92,  0.97),
        "tasa_salud_por_miembro"                : r.uniform(0.0020, 0.0045),
        "tasa_narcomenudeo_por_miembro"         : r.uniform(0.00012, 0.00030),
        "tasa_armas_explosivos_por_miembro"     : r.uniform(0.0030, 0.0075),
        "tasa_financieros_fiscales_por_miembro" : r.uniform(0.0008, 0.0022),
        "tasa_hidrocarburos_por_miembro"        : r.uniform(0.0015, 0.0035),
        "tasa_narcotrafico_por_miembro"         : r.uniform(0.00018, 0.00048),
        "Tasa_reporte_salud"                    : r.uniform(0.70,  0.90),
        "Tasa_reporte_narcomenudeo"             : r.uniform(0.80,  0.95),
        "Tasa_reporte_armas_explosivos"         : r.uniform(0.85,  0.95),
        "Tasa_reporte_financieros_fiscales"     : r.uniform(0.80,  0.95),
        "Tasa_reporte_hidrocarburos"            : r.uniform(0.70,  0.90),
        "Tasa_reporte_narcotrafico"             : r.uniform(0.60,  0.85),
    }


def run_one_scenario(args):
    """Worker puro — sin I/O, sin globals mutables."""
    s, seed, inputs, obs_real, denom_floor, nb_k, lower, upper, cn_total, cn_ext = args
    p   = draw_params(seed)
    rng = np.random.default_rng(seed=seed + 99999)

    if np.isfinite(cn_total):
        c = float(np.clip(1-cn_total, 0.88, 0.99))
        p["Tasa_reporte_homicidio"] = float(rng.uniform(max(c-0.04,0.88), min(c+0.04,0.99)))
        c = float(np.clip(1-cn_total, 0.70, 0.95))
        p["Tasa_reporte_secuestro"] = float(rng.uniform(max(c-0.05,0.70), min(c+0.05,0.95)))
    if np.isfinite(cn_ext):
        c = float(np.clip(1-cn_ext, 0.20, 0.55))
        p["Tasa_reporte_extorsion"] = float(rng.uniform(max(c-0.05,0.20), min(c+0.05,0.55)))

    def obj(x):
        sol = simulate_estado(x[0], x[1], p, inputs)
        if sol is None: return 1e12
        return total_loss(calc_observables(sol, p, inputs), obs_real, denom_floor=denom_floor, nb_k=nb_k)

    from scipy.optimize import minimize
    best = None
    for attempt in range(2):
        rng2 = np.random.default_rng(seed=seed + attempt*777)
        x0   = np.array([rng2.uniform(lower[0], upper[0]),
                         rng2.uniform(lower[1], upper[1])])
        try:
            res = minimize(obj, x0, method="L-BFGS-B",
                           bounds=[(lower[0],upper[0]),(lower[1],upper[1])],
                           options={"maxiter":200,"ftol":1e-9})
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            continue

    if best is None or not np.isfinite(best.fun):
        return None
    sol = simulate_estado(best.x[0], best.x[1], p, inputs)
    if sol is None:
        return None
    obs_pred = calc_observables(sol, p, inputs)
    return {
        "scenario" : s,
        "loss"     : total_loss(obs_pred, obs_real, denom_floor=denom_floor, nb_k=nb_k),
        "A0"       : float(best.x[0]),
        "K0"       : float(best.x[1]),
        "A_series" : sol.y[0].copy(),
        "K_series" : sol.y[1].copy(),
        "obs_pred" : {k: v.copy() for k, v in obs_pred.items()},
    }


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from multiprocessing import Pool, cpu_count
    import time

    N_WORKERS = max(1, cpu_count() - 2)

    # --- Ingesta ---
    def leer_pestana(sheet):
        df = pd.read_excel(RUTA_EXCEL, sheet_name=sheet, dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[df["ENTIDAD"].notna()]
        df = df[~df["ENTIDAD"].str.strip().isin(["", "EXTRANJERO"])]
        df["ANIO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
        df["ENTIDAD"] = df["ENTIDAD"].apply(normalizar_estado)
        df = df[(df["ANIO"] >= ANIO_INICIO) & (df["ANIO"] <= ANIO_FIN)]
        for m in MESES:
            df[m] = pd.to_numeric(df.get(m, 0), errors="coerce").fillna(0)
        return df

    def agregar(df):
        return df.groupby(["ENTIDAD", "ANIO"])[MESES].sum().reset_index()

    def a_serie_larga(df_agg, var):
        rows = []
        for _, row in df_agg.iterrows():
            for i, mes in enumerate(MESES, 1):
                rows.append({
                    "ENTIDAD": row["ENTIDAD"],
                    "ANIO": int(row["ANIO"]),
                    "MES_NUM": i,
                    "T_MES": (int(row["ANIO"]) - ANIO_INICIO) * 12 + i,
                    var: float(row[mes]),
                })
        return pd.DataFrame(rows)

    print("Leyendo datos...")
    sheets_vars = [
        ("HOMICIDIOS", "homicidios"), ("SECUESTRO", "secuestro"),
        ("EXTORSIÓN", "extorsion"),   ("CONTRA LA SALUD", "salud"),
        ("NARCOMENUDEO", "narcomenudeo"), ("FINANCIEROS", "financieros"),
        ("HIDROCARBUROS", "hidrocarburos"), ("TRAFICO", "narcotrafico"),
        ("ARMAS", "armas"),
    ]

    base_cols = ["ENTIDAD", "ANIO", "MES_NUM", "T_MES"]
    datos_delitos = None
    for sheet, var in sheets_vars:
        tmp = a_serie_larga(agregar(leer_pestana(sheet)), var)
        datos_delitos = tmp if datos_delitos is None else datos_delitos.merge(tmp, on=base_cols, how="outer")
    datos_delitos[VARS_DELITO] = datos_delitos[VARS_DELITO].fillna(0)
    datos_delitos = datos_delitos.sort_values(["ENTIDAD", "T_MES"]).reset_index(drop=True)

    cn_raw = pd.read_excel(RUTA_EXCEL, sheet_name="CIFRA NEGRA", dtype=str)
    cn_raw.columns = [c.strip().upper() for c in cn_raw.columns]
    cn_raw = cn_raw[cn_raw["ENTIDAD"].notna()]
    cn_raw["ANIO"] = pd.to_numeric(cn_raw["AÑO"], errors="coerce").astype("Int64")
    cn_raw["ENTIDAD"] = cn_raw["ENTIDAD"].apply(normalizar_estado)
    cn_raw["cn_total"] = pd.to_numeric(cn_raw["TOTAL"], errors="coerce") / 100
    cn_raw["cn_ext"] = pd.to_numeric(cn_raw["EXTORSIÓN"], errors="coerce") / 100
    cn_raw = cn_raw[(cn_raw["ANIO"] >= ANIO_INICIO) & (cn_raw["ANIO"] <= ANIO_FIN)]

    ESTADOS_CANONICOS = sorted(datos_delitos["ENTIDAD"].unique())
    print(f"Estados en datos: {len(ESTADOS_CANONICOS)}")

    # Índice de criminalidad
    W_K = [1.2, 1.1, 1.1, 0.9, 0.6, 0.8, 0.7, 0.7, 0.9]
    base19 = datos_delitos[datos_delitos["ANIO"] == ANIO_INICIO].copy()
    base19["indice"] = sum(w * base19[v] for w, v in zip(W_K, VARS_DELITO))
    indice_df = base19.groupby("ENTIDAD")["indice"].sum().reset_index()
    indice_df["ratio"] = indice_df["indice"] / indice_df["indice"].mean()
    print("\nÍndice de criminalidad relativa por estado (2019):")
    print(indice_df.sort_values("ratio", ascending=False).to_string(index=False))

    # --- Estados a correr ---
    if (not ESTADOS_A_CORRER) or any(str(e).strip().upper() == "TODOS" for e in ESTADOS_A_CORRER):
        estados_finales = ESTADOS_CANONICOS
    else:
        estados_finales = []
        for e in ESTADOS_A_CORRER:
            en = normalizar_estado(e)
            if en in ESTADOS_CANONICOS:
                estados_finales.append(en)
            else:
                print(f"ADVERTENCIA: '{e}' no encontrado.")

    print(f"\nCorrerán {len(estados_finales)} estado(s): {', '.join(estados_finales)}")

    # --- Loop principal ---
    todas_series = []
    todos_rangos = []
    tabla_dic_rows = []

    for estado in estados_finales:
        print(f"\n{'='*52}\nESTADO: {estado}\n{'='*52}")

        obs_e = datos_delitos[datos_delitos["ENTIDAD"] == estado].sort_values("T_MES")
        if obs_e.empty:
            print("  Sin datos. Saltando.")
            continue

        n_meses = int(obs_e["T_MES"].max())
        inputs = make_inputs_estado(n_meses)

        row_r = indice_df[indice_df["ENTIDAD"] == estado]
        ratio = float(np.clip(row_r["ratio"].values[0] if not row_r.empty else 1.0, 0.1, 5.0))
        lower = np.array([500 * ratio, 5000 * ratio])
        upper = np.array([60000 * ratio, 250000 * ratio])
        print(f"  Índice relativo de criminalidad: {ratio:.3f}")

        cn_e = cn_raw[cn_raw["ENTIDAD"] == estado]
        cn_total = float(cn_e["cn_total"].mean()) if not cn_e.empty else np.nan
        cn_ext = float(cn_e["cn_ext"].mean()) if not cn_e.empty else np.nan

        t_idx = obs_e["T_MES"].values - 1
        obs_real = {}
        for v in VARS_DELITO:
            arr = np.zeros(n_meses)
            arr[t_idx] = obs_e[v].values
            obs_real[v] = arr

        # Pre-cálculo de escalas de pérdida por estado
        denom_floor = compute_denom_floor(obs_real)
        nb_k = compute_nb_k(obs_real)

        args_list = [
            (s, 1000 + s, inputs, obs_real, denom_floor, nb_k, lower, upper, cn_total, cn_ext)
            for s in range(N_ESCENARIOS)
        ]

        print(f"  Corriendo {N_ESCENARIOS} escenarios en {N_WORKERS} núcleos...")
        t0 = time.time()
        results_raw = []

        with Pool(processes=N_WORKERS) as pool:
            for i, res in enumerate(pool.imap_unordered(run_one_scenario, args_list), 1):

                if res is not None:
                    results_raw.append(res)

                    A_dic_2025 = float(res["A_series"][-1])  
                    loss_val   = float(res["loss"])

                    print(f"  [{i:4d}/{N_ESCENARIOS}]  A_dic2025={A_dic_2025:,.0f}  loss={loss_val:.4f}")
                else:
                    print(f"  [{i:4d}/{N_ESCENARIOS}]  NO CONVERGIÓ")

        print(f"  Total convergidos: {len(results_raw)}/{N_ESCENARIOS}")

        if not results_raw:
            print("  ADVERTENCIA: ningún escenario convergió.")
            continue

        best = min(results_raw, key=lambda r: r["loss"])

        # ---- Guardar todas las trayectorias A(t) para graficar escenarios ----
        A_all = np.vstack([r["A_series"] for r in results_raw])  
        best  = min(results_raw, key=lambda r: r["loss"])
        A_best = best["A_series"]

        rows = []
        for t_mes in range(1, n_meses + 1):
            row = {
                "ENTIDAD": estado,
                "ANIO": ANIO_INICIO + (t_mes - 1) // 12,
                "MES_NUM": ((t_mes - 1) % 12) + 1,
                "T_MES": t_mes,
                "Miembros_activos_crimen": best["A_series"][t_mes - 1],
                "Recursos_criminales": best["K_series"][t_mes - 1],
                "loss": best["loss"],
            }
            for v in VARS_DELITO:
                row[f"pred_{v}"] = best["obs_pred"][v][t_mes - 1]
            rows.append(row)

        serie_df = pd.DataFrame(rows)
        todas_series.append(serie_df)

        A_avgs = np.array([np.mean(r["A_series"]) for r in results_raw])
        todos_rangos.append({
            "ENTIDAD": estado,
            "A_p10": float(np.percentile(A_avgs, 10)),
            "A_p50": float(np.percentile(A_avgs, 50)),
            "A_p90": float(np.percentile(A_avgs, 90)),
            "A_mejor": float(np.mean(best["A_series"])),
            "loss_mejor": best["loss"],
        })

        tabla_dic_rows.append(
            serie_df[serie_df["MES_NUM"] == 12][["ENTIDAD", "ANIO", "Miembros_activos_crimen"]].copy()
        )

    # --- Resultados y gráficas ---
    if not todas_series:
        print("\nNo hay resultados para mostrar.")
    else:
        df_series = pd.concat(todas_series, ignore_index=True)
        df_rangos = pd.DataFrame(todos_rangos)
        df_dic = pd.concat(tabla_dic_rows, ignore_index=True)
        df_dic_wide = (
            df_dic.pivot(index="ENTIDAD", columns="ANIO", values="Miembros_activos_crimen")
                 .rename(columns=lambda c: f"Dic_{c}")
                 .reset_index()
        )

        print("\n" + "="*60)
        print("TABLA: Miembros activos al cierre de cada año")
        print("="*60)
        print(df_dic_wide.to_string(index=False))

        print("\n" + "="*60)
        print("RANGO DE ESTIMACIONES (Ā promedio del periodo)")
        print("="*60)
        print(df_rangos.to_string(index=False))

        df_series.to_csv("resultados_mensuales_estados.csv", index=False, encoding="utf-8-sig")
        df_dic_wide.to_csv("tabla_diciembre_estados.csv", index=False, encoding="utf-8-sig")
        df_rangos.to_csv("rangos_escenarios_estados.csv", index=False, encoding="utf-8-sig")

        # Gráficas
        plt.rcParams.update({"font.family":"DejaVu Sans",
                              "axes.spines.top":False,"axes.spines.right":False,
                              "axes.grid":True,"grid.color":GREY_LT})
        n_max    = int(df_series["T_MES"].max())
        breaks_t = list(range(1, n_max+1, 12))
        labels_t = [f"Ene\n{y}" for y in range(ANIO_INICIO, ANIO_FIN+1)]
        colores  = [GREEN,NAVY,"#E57373","#FFB300","#7E57C2","#26C6DA"]
        
        fig, ax = plt.subplots(figsize=(12,5))
        for i, df in enumerate(todas_series):
            ax.plot(df["T_MES"], df["Miembros_activos_crimen"],
                    color=colores[i%len(colores)], linewidth=1.8, label=df["ENTIDAD"].iloc[0])
        ax.set_xticks(breaks_t); ax.set_xticklabels(labels_t[:len(breaks_t)])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.set_title("Estimación mensual de miembros activos en crimen organizado",
                     fontweight="bold", color=NAVY)
        ax.set_ylabel("Miembros activos", color=NAVY)
        ax.legend(loc="upper left", framealpha=0.7)
        fig.tight_layout(); fig.savefig("graf_serie_mensual.png", dpi=300); plt.close(fig)
        
        anios_u = sorted(df_dic["ANIO"].unique())
        fig, ax = plt.subplots(figsize=(12,5))
        n_est = len(estados_finales);  bar_w = 0.8/n_est
        for i, est in enumerate(estados_finales):
            sub  = df_dic[df_dic["ENTIDAD"]==est]
            xs   = [j+(i-n_est/2+0.5)*bar_w for j in range(len(anios_u))]
            vals = [float(sub[sub["ANIO"]==a]["Miembros_activos_crimen"].values[0])
                    if len(sub[sub["ANIO"]==a])>0 else 0 for a in anios_u]
            ax.bar(xs, vals, width=bar_w*0.9, color=colores[i%len(colores)], label=est, alpha=0.85)
        ax.set_xticks(range(len(anios_u))); ax.set_xticklabels([str(a) for a in anios_u])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.set_title("Miembros activos al cierre de cada año (diciembre)",
                     fontweight="bold", color=NAVY)
        ax.legend(loc="upper left", framealpha=0.7)
        fig.tight_layout(); fig.savefig("graf_cierre_anual.png", dpi=300); plt.close(fig)
        
        n_del = len(VARS_DELITO);  n_est_g = len(todas_series)
        fig, axes = plt.subplots(n_del, n_est_g, figsize=(5*n_est_g, 3*n_del))
        if n_est_g == 1: axes = axes.reshape(-1,1)
        for j, df in enumerate(todas_series):
            est   = df["ENTIDAD"].iloc[0]
            obs_e = datos_delitos[datos_delitos["ENTIDAD"]==est]
            for i, v in enumerate(VARS_DELITO):
                ax    = axes[i,j]
                obs_m = obs_e.set_index("T_MES")[v].reindex(range(1,n_max+1), fill_value=0)
                ax.bar(obs_m.index, obs_m.values, color=NAVY, alpha=0.7, width=0.8)
                ax.plot(df["T_MES"], df[f"pred_{v}"], color=GREEN, linewidth=1.2)
                ax.set_title(f"{est} — {v}", fontsize=8, color=NAVY)
                ax.set_xticks(breaks_t)
                ax.set_xticklabels(labels_t[:len(breaks_t)], fontsize=6)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        fig.suptitle("Observado vs Predicho", fontweight="bold", color=NAVY)
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig("graf_obs_vs_pred.png", dpi=150); plt.close(fig)
        
    # ============================================================
    # GRÁFICA: TODOS LOS ESCENARIOS (azul) + MEJOR ESCENARIO (verde)
    # ============================================================
    
    # Requiere que existan en memoria:
    #   - results_raw (lista de escenarios convergidos para ese estado)
    #   - best (dict con el mejor escenario)
    #   - n_meses
    #   - estado (string)
    
    if results_raw:
        # Matriz escenarios (cada fila es una trayectoria A(t))
        A_all  = np.vstack([r["A_series"] for r in results_raw])
        A_best = best["A_series"]
    
        # Eje X en meses
        x = np.arange(1, n_meses + 1)
    
        # Etiquetas anuales 
        breaks_t = list(range(1, n_meses + 1, 12))
        labels_t = [f"Ene\n{ANIO_INICIO + (t-1)//12}" for t in breaks_t]
    
        fig, ax = plt.subplots(figsize=(12, 5))
    
        # 1) Todos los escenarios (azul)
        for i in range(A_all.shape[0]):
            ax.plot(x, A_all[i, :], color=NAVY, alpha=0.08, linewidth=1.0)
    
        # 2) Mejor escenario (verde)
        ax.plot(x, A_best, color=GREEN, linewidth=2.8)
    
        ax.set_title(f"Escenarios estimados — {estado}\nEscenarios = azul | Mejor ajuste = verde",
                     fontweight="bold", color=NAVY)
        ax.set_xlabel("Meses", color=NAVY)
        ax.set_ylabel("Miembros activos en organizaciones criminales", color=NAVY)
    
        ax.set_xticks(breaks_t)
        ax.set_xticklabels(labels_t)
    
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    
        fig.tight_layout()
        outname = f"graf_escenarios_vs_mejor_{estado.replace(' ', '_')}.png"
        fig.savefig(outname, dpi=200)
        plt.close(fig)
    
        print(f"  {outname}")
        
        print("\n===== LISTO =====")
        for f in ["resultados_mensuales_estados.csv","tabla_diciembre_estados.csv",
                  "rangos_escenarios_estados.csv","graf_serie_mensual.png",
                  "graf_cierre_anual.png","graf_obs_vs_pred.png"]:
            print(f"  {f}")
