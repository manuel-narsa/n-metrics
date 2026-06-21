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
        n_ext1 = m_jueces // 2; n_ext2 = m_jueces - n_ext1
        mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / m_jueces
        var_max = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / m_jueces
        sigma_max = math.sqrt(var_max)
        
        for f in formas:
            s1 = sum(f)
            s2 = sum(x*x for x in f)
            mu_local = s1 / m_jueces
            var_local = (s2 / m_jueces) - (mu_local * mu_local)
            sigma_local = math.sqrt(max(0.0, var_local))
            agrupacion[f] = max(0.0, 1.0 - (sigma_local / sigma_max))

    elif "Nominal" in topologia:
        max_c = m_jueces * (m_jueces - 1) / 2
        for f in formas:
            counts = [f.count(x) for x in range(1, k_escala + 1)]
            c = sum(v*(v-1)/2 for v in counts)
            agrupacion[f] = c / max_c if max_c > 0 else 0.0
            
    else: # Ordinal
        for f in formas:
            counts = [f.count(x) for x in range(1, k_escala + 1)]
            c = sum(v*(v-1)/2 for v in counts)
            agrupacion[f] = c / (m_jueces * (m_jueces - 1) / 2) if m_jueces > 1 else 0.0

    formas_list = list(agrupacion.keys())
    acuerdos = np.array(list(agrupacion.values()))
    
    n_obj = max(0.0, min(1.0, target_N))
    target_A = n_obj**2
    
    distancias = np.abs(acuerdos - target_A)
    sigma = 0.05 
    pesos = np.exp(-0.5 * (distancias / sigma)**2)
    suma = np.sum(pesos)
    probs = pesos / suma if suma > 0 else np.ones(len(pesos)) / len(pesos)

    idx_elegidos = np.random.choice(len(formas_list), size=n_sujetos, p=probs, replace=True)
    
    matriz = []
    for idx in idx_elegidos:
        forma = np.array(formas_list[idx])
        np.random.shuffle(forma)
        matriz.append(forma)
        
    return np.array(matriz)

def ejecutar_auditoria_cobertura(topologia, dim_n, dim_m, k_escala, target_N, n_experimentos, replicas, estimadores):
    """Simulación Monte Carlo optimizada termodinámicamente en O(1)"""
    stats = defaultdict(lambda: {"hits_pob": 0, "hits_mue": 0, "anchos": [], "pobs": [], "muestras": []})
    
    def _registrar_stat(nombre, mue, inf, sup, pob):
        if mue is None or inf is None or sup is None or pob is None: return
        if np.isnan(mue) or np.isnan(inf) or np.isnan(sup) or np.isnan(pob): return
        
        # 🚀 CORRECCIÓN DEFINITIVA: Blindaje de Coma Flotante a 6 decimales
        # Se redondean los límites para evitar la Caída Topológica en nodos simétricos perfectos
        stats[nombre]["hits_pob"] += 1 if round(inf, 6) <= round(pob, 6) <= round(sup, 6) else 0
        stats[nombre]["hits_mue"] += 1 if round(inf, 6) <= round(mue, 6) <= round(sup, 6) else 0
        
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
                "Media Ancho IC": np.mean(d["anchos"])
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