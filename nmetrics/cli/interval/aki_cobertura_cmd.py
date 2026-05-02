import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

from nmetrics.interval.aki_core import calcular_aki_poblacion_asintotica, calcular_estadisticas_aki
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_aki, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<20} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Estimador", "Cob. Población", "Cob. Muestra", "µ(Población Real)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        print("\n" + "=" * 190)
        print(header)
        print("=" * 190)
    print(fmt_str.format(
        dict_aki['k'], dict_aki['m'], dict_aki['n'], dict_aki['Experimentos'], dict_aki['Réplicas'], dict_aki['Escenario'], dict_aki['Estimador'],
        f"{dict_aki['Cobertura Población (%)']:.1f}%", f"{dict_aki['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_aki['µ(Población Real)']), fmt(dict_aki['µ(Valor Muestra)']),
        fmt(dict_aki['Media IC Inf']), fmt(dict_aki['Media IC Sup']), fmt(dict_aki['Media Ancho IC']),
        f"{dict_aki['Tiempo Medio (s)']:.1f}s"
    ))

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para AKI")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas para Bootstrap (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=1, help="Decimales de ruido paramétrico (default: 1)")
    parser.add_argument("-o", "--output", type=str, default="aki_cobertura.csv", help="Ruta del CSV de salida (default: aki_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    print("=" * 190)
    print(f"🚀 STRESS TEST DE COBERTURA: AKI (BC vs Población Asintótica) - n={args.muestra}")
    print("=" * 190)

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
        pob_vals_reales = []
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='paramétrica', decimales=args.decimales)
            aki_pob_real = calcular_aki_poblacion_asintotica(matriz_empirica, 1, args.escala, multiplicador=1000)
            pob_vals_reales.append(aki_pob_real)
            
            stats_aki = calcular_estadisticas_aki(matriz_empirica, args.replicas)
            
            v_aki = stats_aki['AKI Muestra']
            inf_aki = stats_aki['IC Inf']
            sup_aki = stats_aki['IC Sup']
            
            if not np.isnan(v_aki) and not np.isnan(inf_aki) and not np.isnan(sup_aki):
                vals_mue.append(v_aki)
                infs.append(inf_aki)
                sups.append(sup_aki)
                anchos.append(sup_aki - inf_aki)
                cobs_mue.append(1 if (inf_aki <= v_aki <= sup_aki) else 0)
                cobs_pob.append(1 if (inf_aki <= aki_pob_real <= sup_aki) else 0)

        t_esc = time.time() - t0
        
        res_aki = {
            'k': args.escala, 'm': args.jueces, 'n': args.muestra,
            'Experimentos': args.experimentos, 'Réplicas': args.replicas,
            'Escenario': esc, 'Estimador': 'AKI (BC)',
            'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
            'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
            'µ(Población Real)': safe_mean(pob_vals_reales),
            'µ(Valor Muestra)': safe_mean(vals_mue), 
            'Media IC Inf': safe_mean(infs),
            'Media IC Sup': safe_mean(sups),
            'Media Ancho IC': safe_mean(anchos),
            'Tiempo Medio (s)': t_esc / args.experimentos
        }
        
        imprimir_tabla_consola(res_aki, es_primero=(idx == 0))
        todos_resultados.append(res_aki)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    print(f"\n✅ Análisis completado. Datos guardados en: {args.output}")

if __name__ == "__main__":
    main()