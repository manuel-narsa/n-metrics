import numpy as np
import pandas as pd
import math
import itertools
from collections import defaultdict, Counter

# Importamos los núcleos matemáticos
from nmetrics.interval import ni_core, aki_core, icc21_core
from nmetrics.nominal import nn_core, akn_core, kf_core
from nmetrics.ordinal import no_core, ako_core, w_core

def validar_condiciones_analisis(df, topologia, n_total):
    """Verifica si la matriz tiene el tamaño o formato adecuado para ser evaluada."""
    if df is None or df.empty:
        return False, "La matriz está vacía o es inválida.", True
    if df.shape[1] < 2:
        return False, "Se necesitan al menos 2 jueces (columnas) para calcular consenso.", True
    if n_total < 3:
        return False, "Se recomiendan al menos 3 sujetos para un análisis robusto.", False
    return True, "", False

def generar_matriz_termodinamica_exacta(n_sujetos, m_jueces, k_escala, target_N, topologia):
    """Genera una matriz empírica física (N filas x M columnas)."""
    opciones = range(1, k_escala + 1)                    
    formas = list(itertools.combinations_with_replacement(opciones, m_jueces))
    agrupacion = {} 
    
    if "Intervalar" in topologia:
        n_ext1, n_ext2 = m_jueces // 2, m_jueces - (m_jueces // 2)
        mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / m_jueces
        max_sigma = np.sqrt((n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / m_jueces)
    else: 
        max_coinc = m_jueces * (m_jueces - 1) / 2

    for forma in formas:
        arr = np.array(forma)
        if "Intervalar" in topologia:
            sigma = np.std(arr)
            acuerdo = max(0.0, 1.0 - (sigma / max_sigma)) if max_sigma > 0 else 0.0
            n_val = round(np.sqrt(acuerdo), 6)
        else:
            counts = [np.sum(arr == v) for v in opciones]
            coinc = sum(c * (c - 1) / 2 for c in counts)
            n_val = round(np.sqrt(coinc / max_coinc), 6) if max_coinc > 0 else 0.0
        
        multiplicidad = math.factorial(m_jueces) / np.prod([math.factorial(np.sum(arr == v)) for v in opciones])
        if n_val not in agrupacion: agrupacion[n_val] = []
        agrupacion[n_val].append((forma, multiplicidad))
        
    n_disponibles = sorted(list(agrupacion.keys()))
    n_inf = max([n for n in n_disponibles if n <= target_N]) if any(n <= target_N for n in n_disponibles) else n_disponibles[0]
    n_sup = min([n for n in n_disponibles if n >= target_N]) if any(n >= target_N for n in n_disponibles) else n_disponibles[-1]
    p_sup = (target_N - n_inf) / (n_sup - n_inf) if n_sup != n_inf else 1.0
    
    def samplear(n_val, num):
        if num <= 0: return []
        opcs = agrupacion[n_val]
        probs = np.array([o[1] for o in opcs]) / sum([o[1] for o in opcs])
        return [list(np.random.permutation(opcs[idx][0])) for idx in np.random.choice(len(opcs), size=num, p=probs)]

    filas = samplear(n_sup, int(round(p_sup * n_sujetos))) + samplear(n_inf, n_sujetos - int(round(p_sup * n_sujetos)))
    np.random.shuffle(filas) 
    return np.array(filas)


def ejecutar_auditoria_cobertura(topologia, dim_n, dim_m, k_escala, target_N, n_experimentos, replicas, estimadores):
    """Simulación Monte Carlo optimizada termodinámicamente en O(1)"""
    stats = defaultdict(lambda: {"hits_pob": 0, "hits_mue": 0, "anchos": [], "pobs": [], "muestras": []})
    
    def _registrar_stat(nombre, mue, inf, sup, pob):
        if mue is None or inf is None or sup is None or pob is None: return
        if np.isnan(mue) or np.isnan(inf) or np.isnan(sup) or np.isnan(pob): return
        stats[nombre]["hits_pob"] += 1 if inf <= pob <= sup else 0
        stats[nombre]["hits_mue"] += 1 if inf <= mue <= sup else 0
        stats[nombre]["anchos"].append(sup - inf)
        stats[nombre]["pobs"].append(pob)
        stats[nombre]["muestras"].append(mue)

    for _ in range(n_experimentos):
        matriz = generar_matriz_termodinamica_exacta(dim_n, dim_m, k_escala, target_N, topologia)
        dict_estados = Counter(tuple(x) for x in matriz)

        # 1. Marco N (Cálculo Cuántico O(1))
        try:
            if "Intervalar" in topologia and "NI (Marco N)" in estimadores:
                mue_n, p_real, inf_n, sup_n = ni_core.calcular_estadisticas_ni_unificada(dict_estados, k_escala, replicas)
                _registrar_stat("NI (Marco N)", mue_n, inf_n, sup_n, p_real)
            elif "Nominal" in topologia and "NN (Marco N)" in estimadores:
                mue_n, p_real, inf_n, sup_n = nn_core.calcular_estadisticas_nn_unificada(dict_estados, k_escala, replicas)
                _registrar_stat("NN (Marco N)", mue_n, inf_n, sup_n, p_real)
            elif "Ordinal" in topologia and "NO (Marco N)" in estimadores:
                mue_n, p_real, inf_n, sup_n = no_core.calcular_estadisticas_no_unificada(dict_estados, k_escala, replicas)
                _registrar_stat("NO (Marco N)", mue_n, inf_n, sup_n, p_real)
        except Exception: pass

        # 2. Estimadores Clásicos (Fuerza Bruta)
        if "Intervalar" in topologia:
            if "Alpha Krippendorff" in estimadores:
                try:
                    res_aki = aki_core.calcular_estadisticas_aki(matriz, replicas)
                    pob_aki = aki_core.calcular_aki_poblacion_asintotica(matriz, 1, k_escala, 100)
                    _registrar_stat("Alpha Krippendorff", res_aki['AKI Muestra'], res_aki['IC Inf'], res_aki['IC Sup'], pob_aki)
                except Exception: pass
            
            if "ICC(2,1)" in estimadores:
                try:
                    res_icc = icc21_core.calcular_estadisticas_icc21(matriz)
                    pob_icc = icc21_core.calcular_icc_poblacion_asintotica(matriz, 1, k_escala, 100)
                    _registrar_stat("ICC(2,1)", res_icc['ICC Muestra'], res_icc['IC Inf'], res_icc['IC Sup'], pob_icc)
                except Exception: pass

        elif "Nominal" in topologia:
            if "Alpha Krippendorff" in estimadores:
                try:
                    res_akn = akn_core.calcular_estadisticas_akn(matriz, replicas, k_escala)
                    pob_akn = akn_core.calcular_akn_poblacion_asintotica(matriz, k_escala, 100)
                    _registrar_stat("Alpha Krippendorff", res_akn['AKN Muestra'], res_akn['IC Inf'], res_akn['IC Sup'], pob_akn)
                except Exception: pass
            
            if "Kappa Fleiss" in estimadores:
                try:
                    res_kf = kf_core.calcular_estadisticas_kf(matriz, replicas, k_escala)
                    pob_kf = kf_core.calcular_kf_poblacion_asintotica(matriz, k_escala, 100)
                    _registrar_stat("Kappa Fleiss", res_kf['KF Muestra'], res_kf['IC Inf'], res_kf['IC Sup'], pob_kf)
                except Exception: pass

        else: # Ordinal
            if "Alpha Krippendorff" in estimadores:
                try:
                    res_ako = ako_core.calcular_estadisticas_ako(matriz, replicas, k_escala)
                    pob_ako = ako_core.calcular_ako_poblacion_asintotica(matriz, k_escala, 100)
                    _registrar_stat("Alpha Krippendorff", res_ako['AKO Muestra'], res_ako['IC Inf'], res_ako['IC Sup'], pob_ako)
                except Exception: pass
            
            if "Kendall W" in estimadores:
                try:
                    res_w = w_core.calcular_estadisticas_w(matriz, replicas, k_escala)
                    pob_w = w_core.calcular_w_poblacion_asintotica(matriz, k_escala, 100)
                    _registrar_stat("Kendall W", res_w['W Muestra'], res_w['IC Inf'], res_w['IC Sup'], pob_w)
                except Exception: pass

    # 3. Consolidación de Resultados (Ordenados para que el Marco N salga primero)
    final_res = []
    for mod in sorted(stats.keys(), key=lambda x: 0 if "(Marco N)" in x else 1):
        d = stats[mod]
        if len(d["anchos"]) > 0:
            final_res.append({
                "Estimador": mod,
                "Cob. Población (%)": (d["hits_pob"] / n_experimentos) * 100,
                "Cob. Muestra (%)": (d["hits_mue"] / n_experimentos) * 100,
                "µ(Población Real)": np.mean(d["pobs"]),
                "µ(Valor Muestra)": np.mean(d["muestras"]),
                "Ancho Medio IC": np.mean(d["anchos"])
            })
    return final_res


def ejecutar_duelo_ia(topologia, n, m, k, target_N, n_exp, replicas):
    """Calcula el Error Cuadrático Medio (MSE) frente a TODOS los estimadores clásicos"""
    errores = defaultdict(list)
    
    for _ in range(n_exp):
        matriz = generar_matriz_termodinamica_exacta(n, m, k, target_N, topologia)
        dict_estados = Counter(tuple(x) for x in matriz)
        
        try:
            if "Intervalar" in topologia:
                res_n, p_real, _, _ = ni_core.calcular_estadisticas_ni_unificada(dict_estados, k, replicas)
                errores["NI (Marco N)"].append((res_n - p_real)**2)
                
                try: errores["Alpha Krippendorff"].append((aki_core.calcular_estadisticas_aki(matriz, replicas)['AKI Muestra'] - p_real)**2)
                except: pass
                try: errores["ICC(2,1)"].append((icc21_core.calcular_estadisticas_icc21(matriz)['ICC Muestra'] - p_real)**2)
                except: pass
                
            elif "Nominal" in topologia:
                res_n, p_real, _, _ = nn_core.calcular_estadisticas_nn_unificada(dict_estados, k, replicas)
                errores["NN (Marco N)"].append((res_n - p_real)**2)
                
                try: errores["Alpha Krippendorff"].append((akn_core.calcular_estadisticas_akn(matriz, replicas, k)['AKN Muestra'] - p_real)**2)
                except: pass
                try: errores["Kappa Fleiss"].append((kf_core.calcular_estadisticas_kf(matriz, replicas, k)['KF Muestra'] - p_real)**2)
                except: pass
                
            else: # Ordinal
                res_n, p_real, _, _ = no_core.calcular_estadisticas_no_unificada(dict_estados, k, replicas)
                errores["NO (Marco N)"].append((res_n - p_real)**2)
                
                try: errores["Alpha Krippendorff"].append((ako_core.calcular_estadisticas_ako(matriz, replicas, k)['AKO Muestra'] - p_real)**2)
                except: pass
                try: errores["Kendall W"].append((w_core.calcular_estadisticas_w(matriz, replicas, k)['W Muestra'] - p_real)**2)
                except: pass
        except: continue

    if not errores: return []
    
    final_res = []
    # Ordenamos para que el Marco N siempre sea el primero (referencia)
    for est in sorted(errores.keys(), key=lambda x: 0 if "(Marco N)" in x else 1):
        errs = errores[est]
        if errs:
            final_res.append({
                "Estimador": est, 
                "MSE (Error)": f"{np.mean(errs):.6f}"
            })
            
    return final_res