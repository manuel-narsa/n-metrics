import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

# --- BLINDAJE PARA WINDOWS (Evita error de Emojis) ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Importación absoluta SOLO de la función principal
from nmetrics.nominal.nn_core import calcular_estadisticas_nn
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_nn, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<16} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Motor IC", "Cob. Población", "Cob. Muestra", "µ(Pob. Topológica)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        sys.stdout.buffer.write(("\n" + "=" * len(header) + "\n").encode('utf-8'))
        sys.stdout.buffer.write((header + "\n").encode('utf-8'))
        sys.stdout.buffer.write(("=" * len(header) + "\n").encode('utf-8'))
        sys.stdout.flush()
    
    row = fmt_str.format(
        dict_nn['k'], dict_nn['m'], dict_nn['n'], dict_nn['Experimentos'], dict_nn['Réplicas'], dict_nn['Escenario'], dict_nn['Motor IC'],
        f"{dict_nn['Cobertura Población (%)']:.1f}%", f"{dict_nn['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_nn['µ(Población Real)']), fmt(dict_nn['µ(Valor Muestra)']),
        fmt(dict_nn['Media IC Inf']), fmt(dict_nn['Media IC Sup']), fmt(dict_nn['Media Ancho IC']), f"{dict_nn['Tiempo Medio (s)']:.1f}s"
    )
    sys.stdout.buffer.write((row + "\n").encode('utf-8'))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para NN (Nominal)")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas para SP/BC (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=0, help="Decimales de ruido (default: 0 para Nominal)")
    parser.add_argument("-o", "--output", type=str, default="nn_cobertura.csv", help="Ruta del CSV de salida (default: nn_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    metodos_ic = ['SP', 'BC']
    todos_resultados = []

    sys.stdout.buffer.write(("=" * 190 + "\n").encode('utf-8'))
    sys.stdout.buffer.write((f"🚀 STRESS TEST DE COBERTURA NN: SP vs Bootstrap (BC) - n={args.muestra}\n").encode('utf-8'))
    sys.stdout.buffer.write(("=" * 190 + "\n").encode('utf-8'))
    sys.stdout.flush()

    for idx_esc, esc in enumerate(escenarios):
        matrices_exp = [generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='categórica', decimales=args.decimales) for _ in range(args.experimentos)]
        
        # 1. ESTABLECER LA VERDAD POBLACIONAL (El centro topológico del Marco N)
        pob_vals_reales = []
        for mat in matrices_exp:
            _, nn_pond, *_ = calcular_estadisticas_nn(mat, S_replicas=2, k_escala=args.escala, metodo_ic='SP')
            pob_vals_reales.append(nn_pond)

        for idx_m, metodo in enumerate(metodos_ic):
            t0 = time.time()
            cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
            
            for i in range(args.experimentos):
                # 2. CÁLCULO DEL EXPERIMENTO
                nn_mu, v_nn, inf_nn, sup_nn, *_ = calcular_estadisticas_nn(matrices_exp[i], args.replicas, k_escala=args.escala, metodo_ic=metodo)
                
                if not np.isnan(nn_mu) and not np.isnan(inf_nn) and not np.isnan(sup_nn):
                    vals_mue.append(nn_mu)
                    infs.append(inf_nn)
                    sups.append(sup_nn)
                    anchos.append(sup_nn - inf_nn)
                    
                    # 3. VERIFICACIÓN DE COBERTURAS CRUZADAS
                    cobs_mue.append(1 if (inf_nn <= nn_mu <= sup_nn) else 0)
                    cobs_pob.append(1 if (inf_nn <= pob_vals_reales[i] <= sup_nn) else 0)

            res_nn = {
                'k': args.escala, 'm': args.jueces, 'n': args.muestra,
                'Experimentos': args.experimentos, 'Réplicas': args.replicas,
                'Escenario': esc, 'Motor IC': f"NN ({metodo})",
                'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
                'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
                'µ(Población Real)': safe_mean(pob_vals_reales),
                'µ(Valor Muestra)': safe_mean(vals_mue), 
                'Media IC Inf': safe_mean(infs),
                'Media IC Sup': safe_mean(sups),
                'Media Ancho IC': safe_mean(anchos),
                'Tiempo Medio (s)': (time.time() - t0) / args.experimentos
            }
            imprimir_tabla_consola(res_nn, es_primero=(idx_esc == 0 and idx_m == 0))
            todos_resultados.append(res_nn)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    
    sys.stdout.buffer.write((f"\n✅ Análisis completado. Datos guardados en: {args.output}\n").encode('utf-8'))
    sys.stdout.flush()

if __name__ == "__main__":
    main()