import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from nmetrics.ordinal import no_core, ako_core, w_core
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
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST GLOBAL: Comparativa de Cobertura Ordinal")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas Bootstrap/SP (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=0, help="Decimales (default: 0 para Ordinal)")
    parser.add_argument("-o", "--output", type=str, default="comparativa_ordinal.csv", help="CSV de salida")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.buffer.write((f"🚀 STRESS TEST GLOBAL ORDINAL: NO (SP) vs AKO (BC) vs Kendall W (BC) - n={args.muestra}\n").encode('utf-8'))
    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.flush()

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        data = {
            'NO': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'AKO': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'W': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []}
        }
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='ordinal', decimales=args.decimales)
            
            # --- NO ---
            no_muestra, no_pob_real, inf_no, sup_no, *_ = no_core.calcular_estadisticas_no(matriz_empirica, args.replicas, args.escala, 'SP')
            data['NO']['pobs_reales'].append(no_pob_real)
            if not np.isnan(no_muestra) and not np.isnan(inf_no):
                data['NO']['vals_mue'].append(no_muestra)
                data['NO']['infs'].append(inf_no); data['NO']['sups'].append(sup_no)
                data['NO']['anchos'].append(sup_no - inf_no)
                data['NO']['cobs_mue'].append(1 if (inf_no <= no_muestra <= sup_no) else 0)
                data['NO']['cobs_pob'].append(1 if (inf_no <= no_pob_real <= sup_no) else 0)

            # --- AKO ---
            ako_pob_real = ako_core.calcular_ako_poblacion_asintotica(matriz_empirica, args.escala, multiplicador=100)
            stats_ako = ako_core.calcular_estadisticas_ako(matriz_empirica, args.replicas)
            data['AKO']['pobs_reales'].append(ako_pob_real)
            v_ako, inf_ako, sup_ako = stats_ako['AKO Muestra'], stats_ako['IC Inf'], stats_ako['IC Sup']
            if not np.isnan(v_ako) and not np.isnan(inf_ako):
                data['AKO']['vals_mue'].append(v_ako)
                data['AKO']['infs'].append(inf_ako); data['AKO']['sups'].append(sup_ako)
                data['AKO']['anchos'].append(sup_ako - inf_ako)
                data['AKO']['cobs_mue'].append(1 if (inf_ako <= v_ako <= sup_ako) else 0)
                data['AKO']['cobs_pob'].append(1 if (inf_ako <= ako_pob_real <= sup_ako) else 0)

            # --- Kendall W ---
            w_pob_real = w_core.calcular_w_poblacion_asintotica(matriz_empirica, args.escala, multiplicador=100)
            stats_w = w_core.calcular_estadisticas_w(matriz_empirica, args.replicas)
            data['W']['pobs_reales'].append(w_pob_real)
            v_w, inf_w, sup_w = stats_w['W Muestra'], stats_w['IC Inf'], stats_w['IC Sup']
            if not np.isnan(v_w) and not np.isnan(inf_w):
                data['W']['vals_mue'].append(v_w)
                data['W']['infs'].append(inf_w); data['W']['sups'].append(sup_w)
                data['W']['anchos'].append(sup_w - inf_w)
                data['W']['cobs_mue'].append(1 if (inf_w <= v_w <= sup_w) else 0)
                data['W']['cobs_pob'].append(1 if (inf_w <= w_pob_real <= sup_w) else 0)

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

        resultados_escenario = [agg_resultados('NO', 'NO (SP)', args.replicas), agg_resultados('AKO', 'AKO (Bootstrap C.)', args.replicas), agg_resultados('W', 'Kendall W (Bootstrap C.)', args.replicas)]
        imprimir_tabla_consola(resultados_escenario, es_primero=(idx == 0))
        todos_resultados.extend(resultados_escenario)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=',', decimal='.')
    sys.stdout.buffer.write((f"\n✅ Análisis global completado. Datos guardados en: {args.output}\n").encode('utf-8'))
    sys.stdout.flush()

if __name__ == "__main__":
    main()