import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

from nmetrics.interval.icc21_core import calcular_icc_poblacion_asintotica, calcular_estadisticas_icc21
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_icc, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<20} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Estimador", "Cob. Población", "Cob. Muestra", "µ(Población Real)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        print("\n" + "=" * 190)
        print(header)
        print("=" * 190)
    print(fmt_str.format(
        dict_icc['k'], dict_icc['m'], dict_icc['n'], dict_icc['Experimentos'], dict_icc['Réplicas'], dict_icc['Escenario'], dict_icc['Estimador'],
        f"{dict_icc['Cobertura Población (%)']:.1f}%", f"{dict_icc['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_icc['µ(Población Real)']), fmt(dict_icc['µ(Valor Muestra)']),
        fmt(dict_icc['Media IC Inf']), fmt(dict_icc['Media IC Sup']), fmt(dict_icc['Media Ancho IC']),
        f"{dict_icc['Tiempo Medio (s)']:.1f}s"
    ))

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para ICC(2,1)")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-d", "--decimales", type=int, default=1, help="Decimales de ruido paramétrico (default: 1)")
    parser.add_argument("-o", "--output", type=str, default="icc21_cobertura.csv", help="Ruta del CSV de salida (default: icc21_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    todos_resultados = []

    print("=" * 190)
    print(f"🚀 STRESS TEST DE COBERTURA: ICC(2,1) (F-ANOVA vs Población Asintótica) - n={args.muestra}")
    print("=" * 190)

    for idx, esc in enumerate(escenarios):
        t0 = time.time()
        cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
        pob_vals_reales = []
        
        for i in range(args.experimentos):
            matriz_empirica = generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='paramétrica', decimales=args.decimales)
            icc_pob_real = calcular_icc_poblacion_asintotica(matriz_empirica, 1, args.escala, multiplicador=1000)
            pob_vals_reales.append(icc_pob_real)
            
            stats_icc = calcular_estadisticas_icc21(matriz_empirica)
            
            v_icc = stats_icc['ICC Muestra']
            inf_icc = stats_icc['IC Inf']
            sup_icc = stats_icc['IC Sup']
            
            if not np.isnan(v_icc) and not np.isnan(inf_icc) and not np.isnan(sup_icc):
                vals_mue.append(v_icc)
                infs.append(inf_icc)
                sups.append(sup_icc)
                anchos.append(sup_icc - inf_icc)
                cobs_mue.append(1 if (inf_icc <= v_icc <= sup_icc) else 0)
                cobs_pob.append(1 if (inf_icc <= icc_pob_real <= sup_icc) else 0)

        t_esc = time.time() - t0
        
        res_icc = {
            'k': args.escala, 'm': args.jueces, 'n': args.muestra,
            'Experimentos': args.experimentos, 'Réplicas': "N/A (Analítico)",
            'Escenario': esc, 'Estimador': 'ICC(2,1) [ANOVA]',
            'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
            'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
            'µ(Población Real)': safe_mean(pob_vals_reales),
            'µ(Valor Muestra)': safe_mean(vals_mue), 
            'Media IC Inf': safe_mean(infs),
            'Media IC Sup': safe_mean(sups),
            'Media Ancho IC': safe_mean(anchos),
            'Tiempo Medio (s)': t_esc / args.experimentos
        }
        
        imprimir_tabla_consola(res_icc, es_primero=(idx == 0))
        todos_resultados.append(res_icc)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    print(f"\n✅ Análisis completado. Datos guardados en: {args.output}")

if __name__ == "__main__":
    main()