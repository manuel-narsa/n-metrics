import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

from nmetrics.nominal.akn_core import calcular_akn_poblacion_asintotica, calcular_estadisticas_akn
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_akn, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<20} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Estimador", "Cob. Población", "Cob. Muestra", "µ(Población Real)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        print("\n" + "=" * 190)
        print(header)
        print("=" * 190)
    print(fmt_str.format(
        dict_akn['k'], dict_akn['m'], dict_akn['n'], dict_akn['Experimentos'], dict_akn['Réplicas'], dict_akn['Escenario'], dict_akn['Estimador'],
        f"{dict_akn['Cobertura Población (%)']:.1f}%", f"{dict_akn['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_akn['µ(Población Real)']), fmt(dict_akn['µ(Valor Muestra)']),
        fmt(dict_akn['Media IC Inf']), fmt(dict_akn['Media IC Sup']), fmt(dict_akn['Media Ancho IC']),
        f"{dict_akn['Tiempo Medio (s)']:.1f}s"
    ))

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para AKN (Nominal)")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas para Bootstrap (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=0, help="Decimales de ruido (default: 0 para Nominal)")
    parser.add_argument("-o", "--output", type=str, default="akn_cobertura.csv", help="Ruta del CSV de salida (default: akn_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    print("=" * 190)
    print(f"🚀 STRESS TEST DE COBERTURA: AKN (BC vs Población Asintótica) - n={args.muestra}")
    print("=" * 190)

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
        pob_vals_reales = []
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='categórica', decimales=args.decimales)
            
            akn_pob_real = calcular_akn_poblacion_asintotica(matriz_empirica, k_escala=args.escala, multiplicador=1000)
            pob_vals_reales.append(akn_pob_real)
            
            stats_akn = calcular_estadisticas_akn(matriz_empirica, args.replicas, k_escala=args.escala)
            
            v_akn = stats_akn['AKN Muestra']
            inf_akn = stats_akn['IC Inf']
            sup_akn = stats_akn['IC Sup']
            
            if not np.isnan(v_akn) and not np.isnan(inf_akn) and not np.isnan(sup_akn):
                vals_mue.append(v_akn)
                infs.append(inf_akn)
                sups.append(sup_akn)
                anchos.append(sup_akn - inf_akn)
                cobs_mue.append(1 if (inf_akn <= v_akn <= sup_akn) else 0)
                cobs_pob.append(1 if (inf_akn <= akn_pob_real <= sup_akn) else 0)

        t_esc = time.time() - t0
        
        res_akn = {
            'k': args.escala, 'm': args.jueces, 'n': args.muestra,
            'Experimentos': args.experimentos, 'Réplicas': args.replicas,
            'Escenario': esc, 'Estimador': 'AKN (BC)',
            'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
            'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
            'µ(Población Real)': safe_mean(pob_vals_reales),
            'µ(Valor Muestra)': safe_mean(vals_mue), 
            'Media IC Inf': safe_mean(infs),
            'Media IC Sup': safe_mean(sups),
            'Media Ancho IC': safe_mean(anchos),
            'Tiempo Medio (s)': t_esc / args.experimentos
        }
        
        imprimir_tabla_consola(res_akn, es_primero=(idx == 0))
        todos_resultados.append(res_akn)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    print(f"\n✅ Análisis completado. Datos guardados en: {args.output}")

if __name__ == "__main__":
    main()