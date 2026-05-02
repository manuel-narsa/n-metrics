import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

from nmetrics.ordinal.w_core import calcular_w_poblacion_asintotica, calcular_estadisticas_w
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_w, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<20} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Estimador", "Cob. Población", "Cob. Muestra", "µ(Población Real)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        print("\n" + "=" * 190)
        print(header)
        print("=" * 190)
    print(fmt_str.format(
        dict_w['k'], dict_w['m'], dict_w['n'], dict_w['Experimentos'], dict_w['Réplicas'], dict_w['Escenario'], dict_w['Estimador'],
        f"{dict_w['Cobertura Población (%)']:.1f}%", f"{dict_w['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_w['µ(Población Real)']), fmt(dict_w['µ(Valor Muestra)']),
        fmt(dict_w['Media IC Inf']), fmt(dict_w['Media IC Sup']), fmt(dict_w['Media Ancho IC']),
        f"{dict_w['Tiempo Medio (s)']:.1f}s"
    ))

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para Kendall W")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas para Bootstrap (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=0, help="Decimales de ruido (default: 0 para Ordinal)")
    parser.add_argument("-o", "--output", type=str, default="w_cobertura.csv", help="Ruta del CSV de salida (default: w_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    print("=" * 190)
    print(f"🚀 STRESS TEST DE COBERTURA: Kendall W (BC vs Población Asintótica) - n={args.muestra}")
    print("=" * 190)

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
        pob_vals_reales = []
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='ordinal', decimales=args.decimales)
            
            w_pob_real = calcular_w_poblacion_asintotica(matriz_empirica, k_escala=args.escala, multiplicador=100)
            pob_vals_reales.append(w_pob_real)
            
            stats_w = calcular_estadisticas_w(matriz_empirica, args.replicas, k_escala=args.escala)
            
            v_w = stats_w['W Muestra']
            inf_w = stats_w['IC Inf']
            sup_w = stats_w['IC Sup']
            
            if not np.isnan(v_w) and not np.isnan(inf_w) and not np.isnan(sup_w):
                vals_mue.append(v_w)
                infs.append(inf_w)
                sups.append(sup_w)
                anchos.append(sup_w - inf_w)
                cobs_mue.append(1 if (inf_w <= v_w <= sup_w) else 0)
                cobs_pob.append(1 if (inf_w <= w_pob_real <= sup_w) else 0)

        t_esc = time.time() - t0
        
        res_w = {
            'k': args.escala, 'm': args.jueces, 'n': args.muestra,
            'Experimentos': args.experimentos, 'Réplicas': args.replicas,
            'Escenario': esc, 'Estimador': 'Kendall W (BC)',
            'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
            'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
            'µ(Población Real)': safe_mean(pob_vals_reales),
            'µ(Valor Muestra)': safe_mean(vals_mue), 
            'Media IC Inf': safe_mean(infs),
            'Media IC Sup': safe_mean(sups),
            'Media Ancho IC': safe_mean(anchos),
            'Tiempo Medio (s)': t_esc / args.experimentos
        }
        
        imprimir_tabla_consola(res_w, es_primero=(idx == 0))
        todos_resultados.append(res_w)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    print(f"\n✅ Análisis completado. Datos guardados en: {args.output}")

if __name__ == "__main__":
    main()