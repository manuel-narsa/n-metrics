import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from nmetrics.nominal import nn_core, akn_core, kf_core
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
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST GLOBAL: Comparativa de Cobertura Nominal")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas Bootstrap/SP (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=0, help="Decimales (default: 0 para Nominal)")
    parser.add_argument("-o", "--output", type=str, default="comparativa_nominal.csv", help="CSV de salida")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.buffer.write((f"🚀 STRESS TEST GLOBAL NOMINAL: NN (SP) vs AKN (BC) vs KF (BC) - n={args.muestra}\n").encode('utf-8'))
    sys.stdout.buffer.write(("=" * 195 + "\n").encode('utf-8'))
    sys.stdout.flush()

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        data = {
            'NN': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'AKN': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []},
            'KF': {'cobs_pob': [], 'cobs_mue': [], 'vals_mue': [], 'infs': [], 'sups': [], 'anchos': [], 'pobs_reales': []}
        }
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='categórica', decimales=args.decimales)
            
            # --- NN ---
            nn_muestra, nn_pob_real, inf_nn, sup_nn, *_ = nn_core.calcular_estadisticas_nn(matriz_empirica, args.replicas, args.escala, 'SP')
            data['NN']['pobs_reales'].append(nn_pob_real)
            if not np.isnan(nn_muestra) and not np.isnan(inf_nn):
                data['NN']['vals_mue'].append(nn_muestra)
                data['NN']['infs'].append(inf_nn); data['NN']['sups'].append(sup_nn)
                data['NN']['anchos'].append(sup_nn - inf_nn)
                data['NN']['cobs_mue'].append(1 if (inf_nn <= nn_muestra <= sup_nn) else 0)
                data['NN']['cobs_pob'].append(1 if (inf_nn <= nn_pob_real <= sup_nn) else 0)

            # --- AKN ---
            akn_pob_real = akn_core.calcular_akn_poblacion_asintotica(matriz_empirica, args.escala, multiplicador=100)
            stats_akn = akn_core.calcular_estadisticas_akn(matriz_empirica, args.replicas)
            data['AKN']['pobs_reales'].append(akn_pob_real)
            v_akn, inf_akn, sup_akn = stats_akn['AKN Muestra'], stats_akn['IC Inf'], stats_akn['IC Sup']
            if not np.isnan(v_akn) and not np.isnan(inf_akn):
                data['AKN']['vals_mue'].append(v_akn)
                data['AKN']['infs'].append(inf_akn); data['AKN']['sups'].append(sup_akn)
                data['AKN']['anchos'].append(sup_akn - inf_akn)
                data['AKN']['cobs_mue'].append(1 if (inf_akn <= v_akn <= sup_akn) else 0)
                data['AKN']['cobs_pob'].append(1 if (inf_akn <= akn_pob_real <= sup_akn) else 0)

            # --- KF ---
            kf_pob_real = kf_core.calcular_kf_poblacion_asintotica(matriz_empirica, args.escala, multiplicador=100)
            stats_kf = kf_core.calcular_estadisticas_kf(matriz_empirica, args.replicas)
            data['KF']['pobs_reales'].append(kf_pob_real)
            v_kf, inf_kf, sup_kf = stats_kf['KF Muestra'], stats_kf['IC Inf'], stats_kf['IC Sup']
            if not np.isnan(v_kf) and not np.isnan(inf_kf):
                data['KF']['vals_mue'].append(v_kf)
                data['KF']['infs'].append(inf_kf); data['KF']['sups'].append(sup_kf)
                data['KF']['anchos'].append(sup_kf - inf_kf)
                data['KF']['cobs_mue'].append(1 if (inf_kf <= v_kf <= sup_kf) else 0)
                data['KF']['cobs_pob'].append(1 if (inf_kf <= kf_pob_real <= sup_kf) else 0)

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

        resultados_escenario = [agg_resultados('NN', 'NN (SP)', args.replicas), agg_resultados('AKN', 'AKN (Bootstrap C.)', args.replicas), agg_resultados('KF', 'KF (Bootstrap C.)', args.replicas)]
        imprimir_tabla_consola(resultados_escenario, es_primero=(idx == 0))
        todos_resultados.extend(resultados_escenario)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=',', decimal='.')
    sys.stdout.buffer.write((f"\n✅ Análisis global completado. Datos guardados en: {args.output}\n").encode('utf-8'))
    sys.stdout.flush()

if __name__ == "__main__":
    main()