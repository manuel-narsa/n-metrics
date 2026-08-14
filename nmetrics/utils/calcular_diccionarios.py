from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
import sqlite3
from tqdm.notebook import tqdm

# Núcleo del marco
from nmetrics.interval.ni_core import _build_macrostate_dictionary
from nmetrics.nominal.nn_core import _build_macrostate_dictionary_nn
from nmetrics.ordinal.no_core import _build_macrostate_dictionary_no


def calcular_combinacion(m, k):
  """Calcula las 3 métricas para un par (m, k)."""
  resultados = []

  # 1. NI
  dict_ni = _build_macrostate_dictionary(m, k)
  json_ni = json.dumps(dict_ni, separators=(",", ":"))
  resultados.append(("NI", m, k, json_ni))

  # 2. NN
  dict_nn = _build_macrostate_dictionary_nn(m, k)
  json_nn = json.dumps(dict_nn, separators=(",", ":"))
  resultados.append(("NN", m, k, json_nn))

  # 3. NO
  dict_no = _build_macrostate_dictionary_no(m, k)
  dict_no_json = {str(list(clave)): val for clave, val in dict_no.items()}
  json_no = json.dumps(dict_no_json, separators=(",", ":"))
  resultados.append(("NO", m, k, json_no))

  return resultados


def precargar_cache_inteligente(m_inicio=2, m_fin=100, k_escala=5):
  db_path = "n_metrics_cache.db"

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  cursor.execute("PRAGMA journal_mode = WAL;")
  cursor.execute("PRAGMA synchronous = NORMAL;")

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

  # Separar tareas ligeras (< 60) de pesadas (>= 60)
  m_ligeros = [m for m in range(m_inicio, m_fin + 1) if m < 60]
  m_pesados = [m for m in range(m_inicio, m_fin + 1) if m >= 60]

  # --------------------------------------------------------------------------
  # FASE 1: LIGEROS (Paralelo a máxima velocidad)
  # --------------------------------------------------------------------------
  if m_ligeros:
    print(
        f"⚡ Fase 1: Procesando {len(m_ligeros)} valores de m (<60) en"
        " PARALELO..."
    )
    num_cores = os.cpu_count() or 4
    tareas = [(m, k_escala) for m in m_ligeros]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
      futures = {
          executor.submit(calcular_combinacion, m, k): m for m, k in tareas
      }

      for future in tqdm(
          as_completed(futures),
          total=len(m_ligeros),
          desc="Fase Paralela (m < 60)",
      ):
        m = futures[future]
        try:
          filas = future.result()
          cursor.executemany(
              "INSERT OR REPLACE INTO macroestados_cache VALUES (?, ?, ?, ?)",
              filas,
          )
          conn.commit()
        except Exception as e:
          print(f"Error en m={m}: {e}")

  # --------------------------------------------------------------------------
  # FASE 2: PESADOS (Secuencial seguro con liberación de RAM)
  # --------------------------------------------------------------------------
  if m_pesados:
    print(
        f"\n🛡️ Fase 2: Procesando {len(m_pesados)} valores de m (>=60) en"
        " SECUENCIAL SEGURO..."
    )

    for m in tqdm(m_pesados, desc="Fase Secuencial (m >= 60)"):
      try:
        filas = calcular_combinacion(m, k_escala)
        cursor.executemany(
            "INSERT OR REPLACE INTO macroestados_cache VALUES (?, ?, ?, ?)",
            filas,
        )
        conn.commit()

        # Liberación inmediata de RAM
        del filas
        gc.collect()
      except Exception as e:
        print(f"Error en m={m}: {e}")

  conn.close()
  print(
      f"\n¡Proceso finalizado! Base de datos '{db_path}' actualizada"
      " correctamente."
  )


# Ejemplo de uso: Ejecutar desde m=2 hasta m=120
precargar_cache_inteligente(m_inicio=2, m_fin=120, k_escala=5)