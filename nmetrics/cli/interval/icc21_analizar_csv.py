import argparse
import pandas as pd
import numpy as np
import os
import sys
import time

# Importación relativa dentro del paquete
from nmetrics.interval.icc21_core import calcular_estadisticas_icc21, calcular_icc_poblacion_asintotica

def main():
    parser = argparse.ArgumentParser(description="🔍 ANÁLISIS INDIVIDUAL: Coeficiente de Correlación Intraclase ICC(2,1)")
    parser.add_argument("archivo", help="Ruta al archivo CSV con la matriz empírica")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-s", "--separador", type=str, default=';', help="Separador del CSV (default: ';')")

    args = parser.parse_args()

    if not os.path.exists(args.archivo): 
        print(f"❌ Error: No se encontró el archivo '{args.archivo}'")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.archivo, sep=args.separador, header=None)
        if df.shape[1] == 1 and args.separador == ';':
            df = pd.read_csv(args.archivo, sep=',', header=None)
        df = df.replace(',', '.', regex=True)
        matriz_empirica = df.values.astype(float)
    except Exception as e:
        print(f"❌ Error al procesar el archivo CSV: {e}")
        sys.exit(1)

    print("=" * 115)
    print(f"🚀 ANÁLISIS CLÁSICO PARAMÉTRICO: ICC(2,1) [ANOVA]")
    print("=" * 115)
    
    t0 = time.time()
    print(" -> Calculando estimador analítico de la muestra y F-tests...")
    stats_icc = calcular_estadisticas_icc21(matriz_empirica)
    
    print(" -> Generando universo asintótico (x1000) basado en el Espacio de Configuración...")
    icc_pob_real = calcular_icc_poblacion_asintotica(matriz_empirica, k_min=1, k_max=args.escala, multiplicador=1000)
    t1 = time.time()

    def format_val(val): return "N/A" if pd.isna(val) else f"{val:.4f}"
    
    icc_val = stats_icc['ICC Muestra']
    ic_inf = stats_icc['IC Inf']
    ic_sup = stats_icc['IC Sup']
    ancho_ic = ic_sup - ic_inf if not pd.isna(ic_sup) and not pd.isna(ic_inf) else np.nan

    header = f"| {'Estimador Evaluador':<30} | {'Valor Muestra':>13} | {'Población Teórica':>17} | {'IC Inf':>8} | {'IC Sup':>8} | {'Ancho IC':>8} |"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"| {'ICC(2,1) (Analítico ANOVA)':<30} | {format_val(icc_val):>13} | {format_val(icc_pob_real):>17} | {format_val(ic_inf):>8} | {format_val(ic_sup):>8} | {format_val(ancho_ic):>8} |")
    print("-" * len(header))
    print(f"⏱️ Tiempo de ejecución: {t1 - t0:.2f} segundos")
    print("=" * len(header) + "\n")

if __name__ == "__main__":
    main()