import itertools
import json
import math
import sqlite3
from collections import defaultdict
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import gammaln
import streamlit as st

# ==============================================================================
# ACCESO A CACHÉ Y GENERACIÓN DE MACROESTADOS NI
# ==============================================================================
@st.cache_data(ttl=3600)  
def get_macrostate_dictionary_ni(m_jueces: int, k_escala: int):
    """
    Recupera el diccionario de macroestados desde n_metrics_cache.db.
    """
    try:
        conn = sqlite3.connect("n_metrics_cache.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macroestados_cache (
                metrica TEXT,
                m INTEGER,
                k INTEGER,
                data_json TEXT,
                PRIMARY KEY (metrica, m, k)
            )
        """)
        conn.commit()

        cursor.execute(
            "SELECT data_json FROM macroestados_cache WHERE metrica=? AND m=? AND k=?",
            ("NI", m_jueces, k_escala)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            data = json.loads(row[0])
            return {float(k): v for k, v in data.items()}
            
    except Exception:
        pass

    return _build_macrostate_dictionary(m_jueces, k_escala)

def _build_macrostate_dictionary(m_jueces: int, k_escala: int):
    """Generador algorítmico de macroestados mediante Formas Ancladas (Lógica Original)."""
    fact = [math.factorial(i) for i in range(m_jueces + 1)]

    # Varianza máxima para normalizar el acuerdo
    n_ext1 = m_jueces // 2
    n_ext2 = m_jueces - n_ext1
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / m_jueces
    var_ext = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / m_jueces
    max_sigma = math.sqrt(var_ext)

    macro_dict = defaultdict(float)
    
    # LÓGICA DE FORMAS ANCLADAS:
    # Fijamos el primer elemento como 0 y calculamos combinaciones para el resto.
    # Esto genera todas las estructuras únicas de frecuencias.
    formas_resto = itertools.combinations_with_replacement(range(k_escala), m_jueces - 1)

    for resto in formas_resto:
        forma = (0,) + resto
        max_val = forma[-1]
        
        # Los "desplazamientos" capturan la simetría espacial de la escala.
        # Si la forma llega hasta max_val, puede desplazarse (k_escala - max_val) veces.
        desplazamientos = k_escala - max_val

        # Cálculo exacto de la varianza de esta estructura anclada
        s1 = sum(forma)
        s2 = sum(x * x for x in forma)
        variance = (s2 / m_jueces) - (s1 / m_jueces)**2
        sigma = math.sqrt(max(0, variance))

        acuerdo = max(0.0, 1.0 - (sigma / max_sigma))
        acuerdo_key = round(acuerdo, 8)

        # Multiplicidad combinatoria de los jueces
        counts = {}
        for x in forma:
            counts[x] = counts.get(x, 0) + 1

        denom = 1.0
        for c in counts.values():
            denom *= fact[c]

        multiplicidad_base = fact[m_jueces] / denom
        
        # Peso total en el hiperespacio
        macro_dict[acuerdo_key] += multiplicidad_base * desplazamientos

    return dict(macro_dict)


# ==============================================================================
# CÁLCULOS TERMODINÁMICOS Y PERCENTILES
# ==============================================================================
def calcular_azar_termodinamico_ni(m_jueces: int, k_escala: int = 5):
  macro_dict = get_macrostate_dictionary_ni(m_jueces, k_escala)
  acuerdos = np.array(list(macro_dict.keys()))
  w_topologico = np.array(list(macro_dict.values()))

  sum_w = np.sum(w_topologico)
  mu_global = np.sum(w_topologico * acuerdos) / sum_w
  var_global = np.sum(w_topologico * (acuerdos - mu_global) ** 2) / sum_w
  sigma_global = np.sqrt(var_global)

  return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))


def calcular_percentil_universal_ni(ni_empirico: float, m_jueces: int, k_escala: int = 5):
  macro_dict = get_macrostate_dictionary_ni(m_jueces, k_escala)

  agrupados = defaultdict(float)
  total_espacio = sum(macro_dict.values())

  for acuerdo, multiplicidad_ce in macro_dict.items():
    ni_val = np.sqrt(acuerdo)
    agrupados[round(ni_val, 6)] += multiplicidad_ce

  ordenados = sorted(agrupados.items())
  percentiles = []
  acumulado = 0.0
  for ni_val, mult in ordenados:
    acumulado += mult
    percentil_sup = acumulado / total_espacio
    percentiles.append({'ni': ni_val, 'p_sup': percentil_sup})

  ni_lower = None
  p_lower = None
  ni_upper = None
  p_upper = None

  for p in percentiles:
    if p['ni'] <= ni_empirico:
      ni_lower = p['ni']
      p_lower = p['p_sup']
  for p in reversed(percentiles):
    if p['ni'] >= ni_empirico:
      ni_upper = p['ni']
      p_upper = p['p_sup']

  if ni_lower is None:
    return 0.0
  if ni_upper is None:
    return 100.0
  if ni_lower == ni_upper:
    return p_lower * 100.0

  p_interp = (
      p_lower
      + (p_upper - p_lower)
      * (ni_empirico - ni_lower)
      / (ni_upper - ni_lower)
  )
  return p_interp * 100.0


# ==============================================================================
# DETECCIÓN DE ANOMALÍAS Y POBLACIÓN TEÓRICA
# ==============================================================================
def detectar_anomalias_ni(matriz_entrada, k_escala=5, umbral_sigma=1.0, ni_poblacional=None, modo="infra"):
    """
    Detecta anomalías comparando la frecuencia empírica de cada macroestado
    (vector de frecuencias por categoría) con la esperada bajo azar
    termodinámico en el espacio de configuraciones NI.
    """
    from math import factorial
    from collections import Counter as _C

    X = np.array(matriz_entrada, dtype=float)
    n_total, m = X.shape

    # 1. Cuantizar y calcular firma (vector de frecuencias) de cada sujeto
    X_clip = np.clip(np.floor(X + 0.5), 1, k_escala)
    m_valid = np.sum(~np.isnan(X), axis=1)

    firmas = []
    for row in X_clip:
        # Contar cuántas veces aparece cada categoría (1..k)
        counts_row = _C(int(v) for v in row if not np.isnan(v))
        # Vector de frecuencias ordenado por categoría: (c1, c2, ..., ck)
        firma_vector = tuple(counts_row.get(c, 0) for c in range(1, k_escala + 1))
        firmas.append(firma_vector)

    frec_macro = _C(firmas)
    n_total = len(firmas)

    # 2. Multiplicidad combinatoria Ω de cada firma (vector de frecuencias)
    def _omega_firma(firma):
        # firma es (c1, c2, ..., ck)
        # Ω = m! / (c1! * c2! * ... * ck!)
        denom = 1
        for c in firma:
            denom *= factorial(c)
        return factorial(m) // denom

    omega_macro = {p: _omega_firma(p) for p in frec_macro.keys()}
    total_omega = sum(omega_macro.values()) or 1

    # 3. Distribuciones empírica y azar
    p_emp = {p: frec_macro[p] / n_total for p in frec_macro}
    p_azar = {p: omega_macro[p] / total_omega for p in omega_macro}

    eps = 1e-12
    sobre_rep = {p: (p_emp[p] + eps) / (p_azar[p] + eps) for p in p_emp}

    # 4. NI individual por tupla (α=0 => NI_tupla = sqrt(A_local))
    m_valid_arr = np.sum(~np.isnan(X_clip), axis=1)
    n_ext1 = m_valid_arr // 2
    n_ext2 = m_valid_arr - n_ext1
    safe_m = np.where(m_valid_arr == 0, 1, m_valid_arr)
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / safe_m
    var_ext = (
        n_ext1 * (1 - mean_ext) ** 2 + n_ext2 * (k_escala - mean_ext) ** 2
    ) / safe_m
    max_sigma_row = np.where(m_valid_arr > 1, np.sqrt(var_ext), np.nan)
    sigma_row = np.nanstd(X_clip, axis=1)
    A_local = np.where(
        max_sigma_row > 0, 1.0 - (sigma_row / max_sigma_row), np.nan
    )

    valid = ~np.isnan(A_local)
    if not np.any(valid):
        return pd.DataFrame(), np.nan, np.nan, (np.nan, np.nan)

    NI_tupla = np.sqrt(np.clip(A_local, 0.0, 1.0))

    # 5. Referencia poblacional (diagnóstico)
    if ni_poblacional is None:
        from collections import Counter as _C2
        tuplas = [tuple(int(v) for v in row) for row in X_clip]
        dict_estados = dict(_C2(tuplas))
        _, ni_pob, _, _ = calcular_estadisticas_ni_unificada(
            dict_estados, k_escala, replicas=200
        )
        ni_poblacional = ni_pob

    # 6. Z-score de log10(sobre_rep)
    sobre_rep_arr = np.array([sobre_rep[f] for f in firmas])
    log_sobre = np.log10(sobre_rep_arr + eps)

    media_log = float(np.mean(log_sobre[valid]))
    std_log = float(np.std(log_sobre[valid]))
    if std_log < 1e-12:
        std_log = 1e-12

    z_sobre = (log_sobre - media_log) / std_log

    # Criterio unilateral: solo infra-representación
    if modo == "infra":
        anomalo = valid & (z_sobre < -umbral_sigma)
    elif modo == "supra":
        anomalo = valid & (z_sobre > umbral_sigma)
    else:
        anomalo = valid & (np.abs(z_sobre) > umbral_sigma)

    # 7. DataFrame de resultados
    df_anomalias = pd.DataFrame({
        'Sujeto_ID': np.arange(1, n_total + 1),
        'Acuerdo_Local': A_local,
        'NI_Tupla': NI_tupla,
        'Macroestado': firmas,  # Ahora es un vector (c1, c2, ..., ck)
        'Frecuencia': [frec_macro[f] for f in firmas],
        'P_Empirica': [p_emp[f] for f in firmas],
        'P_Azar': [p_azar[f] for f in firmas],
        'Sobre_Rep': sobre_rep_arr,
        'Z_Sobre': z_sobre,
        'Es_Anomalo': anomalo,
    })

    if modo == "infra":
        z_min, z_max = -umbral_sigma, None
    elif modo == "supra":
        z_min, z_max = None, umbral_sigma
    else:
        z_min, z_max = -umbral_sigma, umbral_sigma

    return df_anomalias, float(ni_poblacional), std_log, (z_min, z_max)

def calcular_estadisticas_ni_unificada(dict_estados, k_escala, replicas=1000):
  estados_lista = list(dict_estados.keys())
  f_t = np.array(list(dict_estados.values()), dtype=float)
  n_total = int(np.sum(f_t))

  X_estados = np.array([list(e) for e in estados_lista], dtype=float)
  num_estados, m_jueces = X_estados.shape

  # 1. ACUERDO ESTATAL
  m_valid = np.sum(~np.isnan(X_estados), axis=1)
  n_ext1 = m_valid // 2
  n_ext2 = m_valid - n_ext1
  safe_m = np.where(m_valid == 0, 1, m_valid)
  mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / safe_m
  var_ext = (
      n_ext1 * (1 - mean_ext) ** 2 + n_ext2 * (k_escala - mean_ext) ** 2
  ) / safe_m
  max_sigma = np.where(m_valid > 1, np.sqrt(var_ext), np.nan)

  sigma_row = np.nanstd(X_estados, axis=1)
  acuerdo = np.where(max_sigma > 0, 1.0 - (sigma_row / max_sigma), 0.0)
  acuerdo = np.nan_to_num(acuerdo, nan=0.0)

  # 2. NI MUESTRA
  w_muestra = f_t / n_total
  mu_m = np.dot(w_muestra, acuerdo)
  var_m = np.maximum(np.dot(w_muestra, acuerdo**2) - mu_m**2, 0.0)
  ni_muestra = np.sqrt(np.maximum(mu_m * (1.0 - np.sqrt(var_m)), 0.0))

  # 3. NI POBLACIÓN
  k_range = np.arange(1, k_escala + 1)
  distancia = np.abs(
      X_estados[:, :, np.newaxis] - k_range[np.newaxis, np.newaxis, :]
  )
  counts = np.nansum(np.maximum(0.0, 1.0 - distancia), axis=1)

  counts_rounded = np.round(counts, 4)
  unique_counts, inverse_idx = np.unique(
      counts_rounded, axis=0, return_inverse=True
  )

  f_M = np.zeros(len(unique_counts))
  np.add.at(f_M, inverse_idx, f_t)
  f_M_mapped = f_M[inverse_idx]

  log_omega = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)
  omega_teorico = np.exp(log_omega - np.max(log_omega))

  w_pob_unnorm = np.where(
      (m_valid > 1) & (f_M_mapped > 0), f_t * (omega_teorico / f_M_mapped), 0.0
  )
  sum_wpob = np.sum(w_pob_unnorm)
  w_pob = (
      w_pob_unnorm / sum_wpob
      if sum_wpob > 0
      else np.ones(num_estados) / num_estados
  )

  mu_p = np.dot(w_pob, acuerdo)
  var_p = np.maximum(np.dot(w_pob, acuerdo**2) - mu_p**2, 0.0)
  ni_poblacion = np.sqrt(np.maximum(mu_p * (1.0 - np.sqrt(var_p)), 0.0))

  # 4. IC SIMULACIÓN
  acuerdos_unicos, inverse_acuerdo = np.unique(
      np.round(acuerdo, 8), return_inverse=True
  )
  w_pob_grouped = np.zeros(len(acuerdos_unicos))
  np.add.at(w_pob_grouped, inverse_acuerdo, w_pob)

  f_boot = np.random.multinomial(n_total, w_pob_grouped, size=replicas)
  w_boot = f_boot / n_total

  mu_sim = np.dot(w_boot, acuerdos_unicos)
  e_x2_sim = np.dot(w_boot, acuerdos_unicos**2)
  var_sim = np.maximum(e_x2_sim - mu_sim**2, 0.0)
  sims = np.sqrt(np.maximum(mu_sim * (1.0 - np.sqrt(var_sim)), 0.0))

  return (
      float(ni_muestra),
      float(ni_poblacion),
      float(np.percentile(sims, 2.5)),
      float(np.percentile(sims, 97.5)),
  )