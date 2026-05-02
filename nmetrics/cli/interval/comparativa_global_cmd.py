import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

# --- BLINDAJE PARA WINDOWS (Evita error de Emojis) ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from nmetrics.interval import ni_core, aki_core, icc21_core
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(resultados_lista, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<22} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Estimador", "Cob. Población", "Cob. Muestra", "µ(Pob. Teórica)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        sys.stdout.buffer.write(("\n" + "=" * 195 + "\n").encode('utf-8'))
        sys.stdout.buffer.write((header + "\n").encode('utf-8'))
        sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
        sys.stdout.flush()
        
    for d in resultados_lista:
        row = fmt_str.format(
            d['k'], d['m'], d['n'], d['Experimentos'], d['Réplicas'], d['Escenario'], d['Estimador'],
            f"{d['Cobertura Población (%)']:.1f}%", f"{d['Cobertura Muestra (%)']:.1f}%",
            fmt(d['µ(Población Real)']), fmt(d['µ(Valor Muestra)']),
            fmt(d['Media IC Inf']), fmt(d['Media IC Sup']), fmt(d['Media Ancho IC']), f"{d['Tiempo Medio (s)']:.1f}s"
        )
        sys.stdout.buffer.write((row + "\n").encode('utf-8'))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST GLOBAL: Comparativa de Cobertura Intervalar")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas Bootstrap/SP (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=1, help="Decimales de ruido paramétrico (default: 1)")
    parser.add_argument("-o", "--output", type=str, default="comparativa_interval.csv", help="CSV de salida")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.buffer.write((f"🚀 STRESS TEST GLOBAL: NI (SP) vs AKI (BC) vs ICC(2,1) (F-ANOVA) - n={args.muestra}\n").encode('utf-8'))
    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.flush()

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        data = {
            'NI': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'AKI': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'ICC21': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []}
        }
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='paramétrica', decimales=args.decimales)
            
            # --- NI ---
            ni_muestra, ni_pob_real, inf_ni, sup_ni, *_ = ni_core.calcular_estadisticas_ni(matriz_empirica, args.replicas, args.escala, 'SP')
            data['NI']['pobs_reales'].append(ni_pob_real)
            if not np.isnan(ni_muestra) and not np.isnan(inf_ni):
                data['NI']['vals_mue'].append(ni_muestra)
                data['NI']['infs'].append(inf_ni); data['NI']['sups'].append(sup_ni)
                data['NI']['anchos'].append(sup_ni - inf_ni)
                data['NI']['cobs_mue'].append(1 if (inf_ni <= ni_muestra <= sup_ni) else 0)
                data['NI']['cobs_pob'].append(1 if (inf_ni <= ni_pob_real <= sup_ni) else 0)

            # --- AKI ---
            aki_pob_real = aki_core.calcular_aki_poblacion_asintotica(matriz_empirica, 1, args.escala, multiplicador=100)
            stats_aki = aki_core.calcular_estadisticas_aki(matriz_empirica, args.replicas)
            data['AKI']['pobs_reales'].append(aki_pob_real)
            v_aki, inf_aki, sup_aki = stats_aki['AKI Muestra'], stats_aki['IC Inf'], stats_aki['IC Sup']
            if not np.isnan(v_aki) and not np.isnan(inf_aki):
                data['AKI']['vals_mue'].append(v_aki)
                data['AKI']['infs'].append(inf_aki); data['AKI']['sups'].append(sup_aki)
                data['AKI']['anchos'].append(sup_aki - inf_aki)
                data['AKI']['cobs_mue'].append(1 if (inf_aki <= v_aki <= sup_aki) else 0)
                data['AKI']['cobs_pob'].append(1 if (inf_aki <= aki_pob_real <= sup_aki) else 0)

            # --- ICC(2,1) ---
            icc_pob_real = icc21_core.calcular_icc_poblacion_asintotica(matriz_empirica, 1, args.escala, multiplicador=100)
            stats_icc = icc21_core.calcular_estadisticas_icc21(matriz_empirica)
            data['ICC21']['pobs_reales'].append(icc_pob_real)
            v_icc, inf_icc, sup_icc = stats_icc['ICC Muestra'], stats_icc['IC Inf'], stats_icc['IC Sup']
            if not np.isnan(v_icc) and not np.isnan(inf_icc):
                data['ICC21']['vals_mue'].append(v_icc)
                data['ICC21']['infs'].append(inf_icc); data['ICC21']['sups'].append(sup_icc)
                data['ICC21']['anchos'].append(sup_icc - inf_icc)
                data['ICC21']['cobs_mue'].append(1 if (inf_icc <= v_icc <= sup_icc) else 0)
                data['ICC21']['cobs_pob'].append(1 if (inf_icc <= icc_pob_real <= sup_icc) else 0)

        t_esc = time.time() - t0
        def agg_resultados(key, nombre, rep_lbl):
            d = data[key]
            return {
                'k': args.escala, 'm': args.jueces, 'n': args.muestra, 'Experimentos': args.experimentos, 'Réplicas': rep_lbl,
                'Escenario': esc, 'Estimador': nombre,
                'Cobertura Población (%)': safe_mean(d['cobs_pob']) * 100, 'Cobertura Muestra (%)': safe_mean(d['cobs_mue']) * 100,
                'µ(Población Real)': safe_mean(d['pobs_reales']), 'µ(Valor Muestra)': safe_mean(d['vals_mue']), 
                'Media IC Inf': safe_mean(d['infs']), 'Media IC Sup': safe_mean(d['sups']),
                'Media Ancho IC': safe_mean(d['anchos']), 'Tiempo Medio (s)': t_esc / args.experimentos / 3 
            }

        resultados_escenario = [agg_resultados('NI', 'NI (SP)', args.replicas), agg_resultados('AKI', 'AKI (Bootstrap C.)', args.replicas), agg_resultados('ICC21', 'ICC(2,1) [ANOVA]', "N/A")]
        imprimir_tabla_consola(resultados_escenario, es_primero=(idx == 0))
        todos_resultados.extend(resultados_escenario)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=',', decimal='.')
    sys.stdout.buffer.write((f"\n✅ Análisis global completado. Datos guardados en: {args.output}\n").encode('utf-8'))
    sys.stdout.flush()

if __name__ == "__main__":
    main()