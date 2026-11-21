from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import sqlite3
import numpy as np
import pandas as pd
from scipy.special import gammaln
import streamlit as st

# ==============================================================================
# ACCESO A CACHÉ Y GENERACIÓN DE MACROESTADOS NN
# ==============================================================================
@st.cache_data(ttl=3600)  
def get_macrostate_dictionary_nn(m_jueces: int, k_escala: int):
  try:
    conn = sqlite3.connect("n_metrics_cache.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT data_json FROM macroestados_cache WHERE metrica=? AND m=? AND k=?",
        ("NN", m_jueces, k_escala),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
      data = json.loads(row[0])
      return {float(k): v for k, v in data.items()}
  except Exception:
    pass

  return _build_macrostate_dictionary_nn(m_jueces, k_escala)

# Añadimos lru_cache para evitar recálculos
@lru_cache(maxsize=128)
def _build_macrostate_dictionary_nn(m_jueces: int, k_escala: int):
    from math import lgamma, exp, log

    max_coincidencias = (m_jueces * (m_jueces - 1)) / 2
    log_fact_m = lgamma(m_jueces + 1)

    def _get_bounded_partitions(n, max_len, max_val=None):
        if max_val is None:
            max_val = n
        if n == 0:
            yield ()
            return
        if max_len == 0:
            return
        for i in range(min(n, max_val), 0, -1):
            for p in _get_bounded_partitions(n - i, max_len - 1, i):
                yield (i,) + p

    log_macro = defaultdict(list)

    for p in _get_bounded_partitions(m_jueces, k_escala):
        v = len(p)
        coincidencias = sum((c * (c - 1)) >> 1 for c in p)
        acuerdo = (round(coincidencias / max_coincidencias, 8) if max_coincidencias > 0 else 0.0)

        log_denom_jueces = 0.0
        for c in p:
            log_denom_jueces += lgamma(c + 1)
        log_mult_base = log_fact_m - log_denom_jueces

        counts_of_sizes = {}
        for c in p:
            counts_of_sizes[c] = counts_of_sizes.get(c, 0) + 1

        log_peso = 0.0
        for i in range(v):
            log_peso += log(k_escala - i)
        for freq in counts_of_sizes.values():
            log_peso -= lgamma(freq + 1)

        log_macro[acuerdo].append(log_mult_base + log_peso)

    log_total = {}
    for acuerdo, logs in log_macro.items():
        m_max = max(logs)
        log_total[acuerdo] = m_max + log(sum(exp(l - m_max) for l in logs))

    max_log = max(log_total.values())
    macro_dict = {a: exp(l - max_log) for a, l in log_total.items()}

    return dict(macro_dict)

# ==============================================================================
# MOTOR TERMODINÁMICO DE SIMULACIÓN PONDERADA
# ==============================================================================
def detectar_anomalias_nn(matriz_entrada, k_escala=5, umbral_sigma=1.0, nn_poblacional=None, modo="infra"):
    """
    Detecta anomalías comparando la frecuencia empírica de cada macroestado
    con la frecuencia esperada bajo azar termodinámico.

    Criterio:
      - Sobre-representación: ratio = P_emp / P_azar
      - Un sujeto es anómalo si |log10(ratio) - media_log| > umbral_sigma * sigma_log
      - Por defecto, solo detecta INFRAREPRESENTACIÓN (ratio bajo), que es
        la firma del "sujeto excepción" introducido artificialmente.

    Parámetros extra:
      - modo: 'infra' (por defecto) o 'bilateral'

    Devuelve:
      df_anomalias, nn_poblacional, sigma_log, (z_min, z_max)
    """
    from math import factorial
    from collections import Counter as _C

    X = np.array(matriz_entrada, dtype=float)
    n_total, m = X.shape

    # ------------------------------------------------------------------
    # 1. Cuantizar y calcular firma (macroestado) de cada sujeto
    # ------------------------------------------------------------------
    X_clip = np.clip(np.floor(X + 0.5), 1, k_escala)
    m_valid = np.sum(~np.isnan(X), axis=1)

    firmas = []
    for row in X_clip:
        counts_row = _C(int(v) for v in row if not np.isnan(v))
        firmas.append(tuple(sorted(counts_row.values(), reverse=True)))

    frec_macro = _C(firmas)
    n_total = len(firmas)

    # ------------------------------------------------------------------
    # 2. Multiplicidad combinatoria Ω de cada firma
    #    Ω(firma) = (m! / Π c_i!) · perm(k, b) / Π (freq_c!)
    # ------------------------------------------------------------------
    def _omega_firma(firma):
        b = len(firma)
        denom_partes = 1
        for c in firma:
            denom_partes *= factorial(c)
        perm_cat = 1
        for i in range(b):
            perm_cat *= (k_escala - i)
        denom_repet = 1
        for freq in _C(firma).values():
            denom_repet *= factorial(freq)
        return (factorial(m) // denom_partes) * perm_cat // max(denom_repet, 1)

    omega_macro = {p: _omega_firma(p) for p in frec_macro.keys()}
    total_omega = sum(omega_macro.values()) or 1

    # ------------------------------------------------------------------
    # 3. Distribuciones empírica y azar
    # ------------------------------------------------------------------
    p_emp = {p: frec_macro[p] / n_total for p in frec_macro}
    p_azar = {p: omega_macro[p] / total_omega for p in omega_macro}

    # Ratio de sobre-representación (con suavizado para evitar log(0))
    eps = 1e-12
    sobre_rep = {
        p: (p_emp[p] + eps) / (p_azar[p] + eps)
        for p in p_emp
    }

    # ------------------------------------------------------------------
    # 4. NN individual por tupla (σ=0 → sqrt(A_local))
    # ------------------------------------------------------------------
    counts = np.zeros((n_total, k_escala))
    for k_val in range(1, k_escala + 1):
        counts[:, k_val - 1] = np.sum(X_clip == k_val, axis=1)

    coincidencias = np.sum(counts * (counts - 1) / 2, axis=1)
    max_coinc = m_valid * (m_valid - 1) / 2

    A_local = np.divide(
        coincidencias, max_coinc,
        out=np.full_like(coincidencias, np.nan, dtype=float),
        where=max_coinc > 0,
    )
    valid = ~np.isnan(A_local)
    if not np.any(valid):
        return pd.DataFrame(), np.nan, np.nan, (np.nan, np.nan)

    NN_tupla = np.sqrt(np.clip(A_local, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 5. NN poblacional de referencia (para mostrar en UI)
    # ------------------------------------------------------------------
    if nn_poblacional is None:
        dict_estados = dict(_C(tuple(int(v) for v in row) for row in X_clip))
        _, nn_pob, _, _ = calcular_estadisticas_nn_unificada(
            dict_estados, k_escala, replicas=200
        )
        nn_poblacional = nn_pob

    # ------------------------------------------------------------------
    # 6. Z-score de log10(sobre_rep)
    # ------------------------------------------------------------------
    sobre_rep_arr = np.array([sobre_rep[f] for f in firmas])
    log_sobre = np.log10(sobre_rep_arr + eps)

    media_log = float(np.mean(log_sobre[valid]))
    std_log = float(np.std(log_sobre[valid]))
    if std_log < 1e-12:
        std_log = 1e-12

    z_sobre = (log_sobre - media_log) / std_log

    # Criterio unilateral: solo infrarrepresentación (S5, caos excesivo)
    if modo == "infra":
        anomalo = valid & (z_sobre < -umbral_sigma)
    elif modo == "supra":
        anomalo = valid & (z_sobre > umbral_sigma)
    else:
        anomalo = valid & (np.abs(z_sobre) > umbral_sigma)

    # ------------------------------------------------------------------
    # 7. DataFrame de resultados
    # ------------------------------------------------------------------
    df_anomalias = pd.DataFrame({
        'Sujeto_ID': np.arange(1, n_total + 1),
        'Acuerdo_Local': A_local,
        'NN_Tupla': NN_tupla,
        'Macroestado': firmas,
        'Frecuencia': [frec_macro[f] for f in firmas],
        'P_Empirica': [p_emp[f] for f in firmas],
        'P_Azar': [p_azar[f] for f in firmas],
        'Sobre_Rep': sobre_rep_arr,
        'Z_Sobre': z_sobre,
        'Es_Anomalo': anomalo,
    })

    # Límites en escala log10 para UI
    if modo == "infra":
        z_min, z_max = -umbral_sigma, None
    elif modo == "supra":
        z_min, z_max = None, umbral_sigma
    else:
        z_min, z_max = -umbral_sigma, umbral_sigma

    return df_anomalias, float(nn_poblacional), std_log, (z_min, z_max)

def calcular_azar_termodinamico_nn(m_jueces, k_escala=5):
  macro_dict = get_macrostate_dictionary_nn(m_jueces, k_escala)
  acuerdos = np.array(list(macro_dict.keys()))
  w_topologico = np.array(list(macro_dict.values()))

  sum_w = np.sum(w_topologico)
  mu_global = np.sum(w_topologico * acuerdos) / sum_w
  var_global = np.sum(w_topologico * (acuerdos - mu_global) ** 2) / sum_w
  sigma_global = np.sqrt(var_global)

  return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))


def calcular_percentil_universal_nn(nn_empirico, m_jueces, k_escala=5):
  macro_dict = get_macrostate_dictionary_nn(m_jueces, k_escala)
  agrupados = defaultdict(float)
  total_espacio = sum(macro_dict.values())

  for acuerdo, multiplicidad_ce in macro_dict.items():
    nn_val = np.sqrt(acuerdo)
    agrupados[round(nn_val, 6)] += multiplicidad_ce

  ordenados = sorted(agrupados.items())
  percentiles = []
  acumulado = 0.0
  for nn_val, mult in ordenados:
    acumulado += mult
    percentil_sup = acumulado / total_espacio
    percentiles.append({'nn': nn_val, 'p_sup': percentil_sup})

  nn_lower, p_lower, nn_upper, p_upper = None, None, None, None

  for p in percentiles:
    if p['nn'] <= nn_empirico:
      nn_lower = p['nn']
      p_lower = p['p_sup']
    else:
      break

  for p in reversed(percentiles):
    if p['nn'] >= nn_empirico:
      nn_upper = p['nn']
      p_upper = p['p_sup']
    else:
      break

  if nn_lower is None: return 0.0
  if nn_upper is None: return 100.0
  if nn_lower == nn_upper: return p_lower * 100.0

  p_interp = p_lower + (p_upper - p_lower) * (nn_empirico - nn_lower) / (nn_upper - nn_lower)
  return p_interp * 100.0


def calcular_poblacion_real_teorica(m_jueces, k_escala=5):
  macro_dict = get_macrostate_dictionary_nn(m_jueces, k_escala)
  acuerdos = np.array(list(macro_dict.keys()))
  multiplicidades = np.array(list(macro_dict.values()))

  mu_poblacional = np.sum(acuerdos * multiplicidades) / np.sum(multiplicidades)
  return float(np.sqrt(mu_poblacional))


# ==============================================================================
# MOTOR DE INFERENCIA UNIFICADA
# ==============================================================================
def calcular_estadisticas_nn_unificada(dict_estados, k_escala, replicas=1000):
  estados_lista = list(dict_estados.keys())
  f_t = np.array(list(dict_estados.values()), dtype=float)
  n_total = int(np.sum(f_t))

  try:
    X_estados = np.array([list(e) for e in estados_lista], dtype=float)
  except ValueError:
    raise ValueError('La matriz contiene datos no numéricos.')

  U, m = X_estados.shape

  m_valid = np.sum(~np.isnan(X_estados), axis=1)
  max_coincidencias = m_valid * (m_valid - 1) / 2

  counts = np.zeros((U, k_escala))
  X_clean = np.floor(X_estados + 0.5)

  for i in range(U):
    row_valid = X_clean[i][~np.isnan(X_clean[i])]
    if len(row_valid) > 0:
      _, freqs = np.unique(row_valid, return_counts=True)
      counts[i, : len(freqs)] = freqs

  coincidencias = np.sum(counts * (counts - 1) / 2, axis=1)

  acuerdo = np.divide(
      coincidencias,
      max_coincidencias,
      out=np.zeros_like(coincidencias, dtype=float),
      where=max_coincidencias > 0,
  )
  acuerdo = np.nan_to_num(acuerdo, nan=0.0)

  def _nn_desde_pesos(w):
    if w.ndim == 1:
      w = w[None, :]
    mu = np.sum(acuerdo * w, axis=1)
    var = np.sum(w * (acuerdo - mu[:, None]) ** 2, axis=1)
    return np.sqrt(np.maximum(mu * (1.0 - np.sqrt(var)), 0.0))

  # 1. NN MUESTRA
  w_muestra = f_t / n_total
  w_muestra = w_muestra / np.sum(w_muestra)
  w_muestra[-1] = 1.0 - np.sum(w_muestra[:-1])
  w_muestra = np.clip(w_muestra, 0.0, 1.0)

  nn_muestra = _nn_desde_pesos(w_muestra)[0]

  # 2. NN POBLACIÓN
  counts_rounded = np.round(counts, 4)
  counts_sorted = np.sort(counts_rounded, axis=1)[:, ::-1]
  unique_counts, inverse_idx = np.unique(counts_sorted, axis=0, return_inverse=True)

  f_M = np.zeros(len(unique_counts))
  np.add.at(f_M, inverse_idx, f_t)
  f_M_mapped = f_M[inverse_idx]

  log_perm_jueces = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)

  f_c = np.zeros((U, m + 1))
  for c in range(m + 1):
    f_c[:, c] = np.sum(counts == c, axis=1)

  log_asign_cat = gammaln(k_escala + 1) - np.sum(gammaln(f_c + 1), axis=1)

  log_omega = log_perm_jueces + log_asign_cat
  omega_teorico = np.exp(log_omega - np.max(log_omega))

  w_pob = np.where((m_valid > 1) & (f_M_mapped > 0), f_t * (omega_teorico / f_M_mapped), 0.0)
  sum_wpob = np.sum(w_pob)

  if sum_wpob > 0:
    w_pob = w_pob / sum_wpob
  else:
    w_pob = np.ones_like(w_pob) / len(w_pob)

  w_pob[-1] = 1.0 - np.sum(w_pob[:-1])
  w_pob = np.clip(w_pob, 0.0, 1.0)

  nn_poblacion = _nn_desde_pesos(w_pob)[0]

  # 3. IC SIMULACIÓN
  safe_n = max(n_total, 2)
  acuerdos_unicos, inverse_acuerdo = np.unique(np.round(acuerdo, 8), return_inverse=True)
  w_pob_grouped = np.zeros(len(acuerdos_unicos))
  np.add.at(w_pob_grouped, inverse_acuerdo, w_pob)

  def _nn_desde_pesos_grouped(w_grouped):
    if w_grouped.ndim == 1:
      w_grouped = w_grouped[None, :]
    mu = np.sum(acuerdos_unicos * w_grouped, axis=1)
    var = np.sum(w_grouped * (acuerdos_unicos - mu[:, None]) ** 2, axis=1)
    return np.sqrt(np.maximum(mu * (1.0 - np.sqrt(var)), 0.0))

  f_boot = np.random.multinomial(safe_n, w_pob_grouped, size=replicas)
  w_boot = f_boot / safe_n

  sims = _nn_desde_pesos_grouped(w_boot)

  return (
      float(nn_muestra),
      float(nn_poblacion),
      float(np.percentile(sims, 2.5)),
      float(np.percentile(sims, 97.5)),
  )