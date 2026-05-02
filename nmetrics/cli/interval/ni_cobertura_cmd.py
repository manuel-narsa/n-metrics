import argparse
import pandas as pd
import numpy as np
import os
import time
import sys

# --- BLINDAJE PARA WINDOWS (Evita error de Emojis) ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Importaciones absolutas del paquete
from nmetrics.interval.ni_core import calcular_estadisticas_ni
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

def fmt(v):
    if pd.isna(v): return "N/A"
    return f"{v:.4f}".replace('.', ',') if isinstance(v, (float, np.float64)) else str(v)

def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else np.nan

def imprimir_tabla_consola(dict_ni, es_primero):
    fmt_str = "| {:^2} | {:^2} | {:^3} | {:^4} | {:^4} | {:<15} | {:<16} | {:>14} | {:>14} | {:>17} | {:>16} | {:>9} | {:>9} | {:>11} | {:>8} |"
    if es_primero:
        header = fmt_str.format("k", "m", "n", "Exp", "Rep", "Escenario", "Motor IC", "Cob. Población", "Cob. Muestra", "µ(Pob. Topológica)", "µ(Valor Muestra)", "µ(IC Inf)", "µ(IC Sup)", "µ(Ancho IC)", "Tiempo")
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
    print(fmt_str.format(
        dict_ni['k'], dict_ni['m'], dict_ni['n'], dict_ni['Experimentos'], dict_ni['Réplicas'], dict_ni['Escenario'], dict_ni['Motor IC'],
        f"{dict_ni['Cobertura Población (%)']:.1f}%", f"{dict_ni['Cobertura Muestra (%)']:.1f}%",
        fmt(dict_ni['µ(Población Real)']), fmt(dict_ni['µ(Valor Muestra)']),
        fmt(dict_ni['Media IC Inf']), fmt(dict_ni['Media IC Sup']), fmt(dict_ni['Media Ancho IC']), f"{dict_ni['Tiempo Medio (s)']:.1f}s"
    ))

def main():
    parser = argparse.ArgumentParser(description="🚀 STRESS TEST: Análisis de Cobertura para NI")
    parser.add_argument("-n", "--muestra", type=int, default=50, help="Tamaño de la muestra (Sujetos) (default: 50)")
    parser.add_argument("-m", "--jueces", type=int, default=7, help="Número de jueces (default: 7)")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-e", "--experimentos", type=int, default=100, help="Número de experimentos Monte Carlo (default: 100)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Número de réplicas para SP/BC (default: 1000)")
    parser.add_argument("-d", "--decimales", type=int, default=1, help="Decimales de ruido paramétrico (default: 1)")
    parser.add_argument("-o", "--output", type=str, default="ni_cobertura.csv", help="Ruta del CSV de salida (default: ni_cobertura.csv)")

    args = parser.parse_args()

    escenarios = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
    metodos_ic = ['SP', 'BC']
    todos_resultados = []

    print("=" * 190)
    # Forzamos la impresión en bytes UTF-8 para evitar corrupción en Jupyter/Windows
    sys.stdout.buffer.write(f"🚀 STRESS TEST DE COBERTURA NI: SP vs Bootstrap (BC) - n={args.muestra}\n".encode('utf-8'))
    sys.stdout.flush()
    print("=" * 190)

    for idx_esc, esc in enumerate(escenarios):
        matrices_exp = [generar_matriz_dinamica(args.muestra, args.jueces, args.escala, esc, tipo_escala='paramétrica', decimales=args.decimales) for _ in range(args.experimentos)]
        
        # 1. ESTABLECER LA VERDAD POBLACIONAL (El centro topológico del Marco N)
        pob_vals_reales = []
        for mat in matrices_exp:
            # En el Marco N, la población teórica es la Esperanza Ponderada (ni_ponderado)
            _, ni_pond, *_ = calcular_estadisticas_ni(mat, S_replicas=2, k_escala=args.escala, metodo_ic='SP')
            pob_vals_reales.append(ni_pond)

        for idx_m, metodo in enumerate(metodos_ic):
            t0 = time.time()
            cobs_pob, cobs_mue, vals_mue, infs, sups, anchos = [], [], [], [], [], []
            
            for i in range(args.experimentos):
                # 2. CÁLCULO DEL EXPERIMENTO
                ni_mu, v_ni, inf_ni, sup_ni, *_ = calcular_estadisticas_ni(matrices_exp[i], args.replicas, k_escala=args.escala, metodo_ic=metodo)
                
                if not np.isnan(ni_mu) and not np.isnan(inf_ni) and not np.isnan(sup_ni):
                    # Guardamos SIEMPRE el valor de la muestra empírica cruda para comparar
                    vals_mue.append(ni_mu)
                    infs.append(inf_ni)
                    sups.append(sup_ni)
                    anchos.append(sup_ni - inf_ni)
                    
                    # 3. VERIFICACIÓN DE COBERTURAS CRUZADAS
                    # Cobertura Muestra: ¿El IC atrapa al valor sesgado empírico (ni_mu)?
                    cobs_mue.append(1 if (inf_ni <= ni_mu <= sup_ni) else 0)
                    
                    # Cobertura Población: ¿El IC atrapa a la Verdad Termodinámica (pob_vals_reales)?
                    cobs_pob.append(1 if (inf_ni <= pob_vals_reales[i] <= sup_ni) else 0)

            res_ni = {
                'k': args.escala, 'm': args.jueces, 'n': args.muestra,
                'Experimentos': args.experimentos, 'Réplicas': args.replicas,
                'Escenario': esc, 'Motor IC': f"NI ({metodo})",
                'Cobertura Población (%)': safe_mean(cobs_pob) * 100,
                'Cobertura Muestra (%)': safe_mean(cobs_mue) * 100,
                'µ(Población Real)': safe_mean(pob_vals_reales),
                'µ(Valor Muestra)': safe_mean(vals_mue), 
                'Media IC Inf': safe_mean(infs),
                'Media IC Sup': safe_mean(sups),
                'Media Ancho IC': safe_mean(anchos),
                'Tiempo Medio (s)': (time.time() - t0) / args.experimentos
            }
            imprimir_tabla_consola(res_ni, es_primero=(idx_esc == 0 and idx_m == 0))
            todos_resultados.append(res_ni)

    df_export = pd.DataFrame(todos_resultados)
    archivo_existe = os.path.isfile(args.output)
    df_export.to_csv(args.output, mode='a' if archivo_existe else 'w', header=not archivo_existe, index=False, sep=';', decimal=',')
    sys.stdout.buffer.write(f"\n✅ Análisis completado. Datos guardados en: {args.output}\n".encode('utf-8'))
    sys.stdout.flush()

if __name__ == "__main__":
    main()