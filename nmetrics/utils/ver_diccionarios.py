import itertools
import math
import pandas as pd


def reconstruir_espectro_ni(m=7, k=5):
  """Calcula el desglose completo de N_I asignando cada N_local a sus tuplas de origen."""

  def calc_ni_local(comp):
    d_sum = 0.0
    for i in range(k):
      for j in range(i + 1, k):
        # Distancia métrica al cuadrado normalizada por (k-1)^2
        d_ij = ((i - j) / (k - 1)) ** 2
        d_sum += comp[i] * comp[j] * d_ij
    disagreement = (2.0 / (m * (m - 1))) * d_sum
    return round(1.0 - disagreement, 8)

  # Generar todas las composiciones posibles
  composiciones = [
      c for c in itertools.product(range(m + 1), repeat=k) if sum(c) == m
  ]

  resumen = {}
  for comp in composiciones:
    n_loc = calc_ni_local(comp)

    # Multiplicidad multinomial del microestado
    denom = math.prod(math.factorial(x) for x in comp)
    omega_micro = math.factorial(m) // denom

    # Partición ordenada decreciente (Tupla de clase)
    tupla_clase = tuple(sorted(comp, reverse=True))

    if n_loc not in resumen:
      resumen[n_loc] = {
          "Omega_Total": 0,
          "Tuplas_Particion": set(),
          "Ejemplo_Composicion": str(comp),
      }

    resumen[n_loc]["Omega_Total"] += omega_micro
    resumen[n_loc]["Tuplas_Particion"].add(str(tupla_clase))

  # Formatear como DataFrame
  filas = []
  total_micro = k**m
  for n_loc, info in sorted(
      resumen.items(), key=lambda x: x[0], reverse=True
  ):
    prob = info["Omega_Total"] / total_micro
    filas.append({
        "Metrica": "NI",
        "m_Evaluadores": m,
        "k_Categorias": k,
        "N_local": n_loc,
        "Multiplicidad_Omega": info["Omega_Total"],
        "Probabilidad": prob,
        "Tuplas_Particion": ", ".join(sorted(info["Tuplas_Particion"])),
    })

  return pd.DataFrame(filas)


# Ejecutar reconstrucción para m=7, k=5
df_ni_desglosado = reconstruir_espectro_ni(7, 5)
df_ni_desglosado.head(49)