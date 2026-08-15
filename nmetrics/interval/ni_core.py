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
    Si la tabla o la BD no existen, las inicializa automáticamente 
    y recurre al fallback algorítmico sin interrumpir al usuario.
    """
    try:
        conn = sqlite3.connect("n_metrics_cache.db")
        cursor = conn.cursor()
        
        # Crear la tabla automáticamente si no existe en el entorno
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

        # Buscar el registro solicitado
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
        # Si ocurre cualquier anomalía con SQLite, pasa silenciosamente al fallback
        pass

    # Fallback: Cálculo en vivo mediante Formas Ancladas si no está en caché
    return _build_macrostate_dictionary(m_jueces, k_escala)
# SE ELIMINÓ EL DECORADOR @lru_cache(maxsize=128) AQUÍ PARA LIBERAR RAM
def _build_macrostate_dictionary(m_jueces: int, k_escala: int):
    """Generador algorítmico de macroestados mediante Formas Ancladas."""
    fact = [math.factorial(i) for i in range(m_jueces + 1)]

    n_ext1 = m_jueces // 2
    n_ext2 = m_jueces - n_ext1
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / m_jueces
    var_ext = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / m_jueces
    max_sigma = math.sqrt(var_ext)

    macro_dict = defaultdict(float)
    formas_resto = itertools.combinations_with_replacement(range(k_escala), m_jueces - 1)

    for resto in formas_resto:
        forma = (0,) + resto
        max_val = forma[-1]
        desplazamientos = k_escala - max_val

        s1 = sum(forma)
        s2 = sum(x * x for x in forma)
        variance = (s2 / m_jueces) - (s1 / m_jueces)**2
        sigma = math.sqrt(max(0, variance))

        acuerdo = max(0.0, 1.0 - (sigma / max_sigma))
        acuerdo_key = round(acuerdo, 8)

        counts = {}
        for x in forma:
            counts[x] = counts.get(x, 0) + 1

        denom = 1.0
        for c in counts.values():
            denom *= fact[c]

        multiplicidad_base = fact[m_jueces] / denom
        macro_dict[acuerdo_key] += multiplicidad_base * desplazamientos

    return dict(macro_dict)


# ==============================================================================
# CÁLCULOS TERMODINÁMICOS Y PERCENTILES
# ==============================================================================


def calcular_azar_termodinamico_ni(m_jueces: int, k_escala: int = 5):
  # CAMBIO CLAVE: Usamos la función optimizada con lectura de caché
  macro_dict = get_macrostate_dictionary_ni(m_jueces, k_escala)

  acuerdos = np.array(list(macro_dict.keys()))
  w_topologico = np.array(list(macro_dict.values()))

  sum_w = np.sum(w_topologico)
  mu_global = np.sum(w_topologico * acuerdos) / sum_w
  var_global = np.sum(w_topologico * (acuerdos - mu_global) ** 2) / sum_w
  sigma_global = np.sqrt(var_global)

  return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))


def calcular_percentil_universal_ni(
    ni_empirico: float, m_jueces: int, k_escala: int = 5
):
  # CAMBIO CLAVE: Usamos la función optimizada con lectura de caché
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


def detectar_anomalias_ni(matriz_entrada, k_escala=5, umbral_sigma=1.0):
  X = np.array(matriz_entrada, dtype=float)
  m_valid = np.sum(~np.isnan(X), axis=1)
  n_ext1 = m_valid // 2
  n_ext2 = m_valid - n_ext1
  safe_m = np.where(m_valid == 0, 1, m_valid)
  mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / safe_m
  var_ext = (
      n_ext1 * (1 - mean_ext) ** 2 + n_ext2 * (k_escala - mean_ext) ** 2
  ) / safe_m
  max_sigma_row = np.where(m_valid > 1, np.sqrt(var_ext), np.nan)
  sigma_row = np.nanstd(X, axis=1)
  acuerdo_local = np.where(
      max_sigma_row > 0, 1.0 - (sigma_row / max_sigma_row), np.nan
  )

  valid = ~np.isnan(acuerdo_local)
  if not np.any(valid):
    return pd.DataFrame(), np.nan, np.nan, np.nan
  mu_global = np.mean(acuerdo_local[valid])
  sigma_global = np.std(acuerdo_local[valid])
  limite = mu_global - (umbral_sigma * sigma_global)
  anomalo = (acuerdo_local < limite) & valid
  df_anomalias = pd.DataFrame({
      'Sujeto_ID': np.arange(1, len(X) + 1),
      'Acuerdo_Local': acuerdo_local,
      'Es_Anomalo': anomalo,
  })
  return df_anomalias, mu_global, sigma_global, limite


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