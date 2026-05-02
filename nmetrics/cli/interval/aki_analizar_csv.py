import argparse
import pandas as pd
import numpy as np
import os
import sys
import time

# Importación relativa dentro del paquete
from  nmetrics.interval.aki_core import calcular_estadisticas_aki, calcular_aki_poblacion_asintotica

def main():
    parser = argparse.ArgumentParser(description="🔍 ANÁLISIS INDIVIDUAL: Alpha de Krippendorff Intervalar (AKI)")
    parser.add_argument("archivo", help="Ruta al archivo CSV con la matriz empírica")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-r", "--replicas", type=int, default=5000, help="Número de réplicas Bootstrap (default: 5000)")
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
    print(f"🚀 ANÁLISIS DEL COEFICIENTE AKI (Alpha de Krippendorff y Bootstrap Clásico)")
    print("=" * 115)
    
    t0 = time.time()
    print(f" -> Generando y evaluando universo de réplicas BC ({args.replicas} iteraciones)...")
    stats_aki = calcular_estadisticas_aki(matriz_empirica, args.replicas)

    print(" -> Generando universo asintótico (x1000) basado en el Espacio de Configuración...")
    aki_pob_real = calcular_aki_poblacion_asintotica(matriz_empirica, k_min=1, k_max=args.escala, multiplicador=1000)
    t1 = time.time()

    def format_val(val): return "N/A" if pd.isna(val) else f"{val:.4f}"
    
    aki_val = stats_aki['AKI Muestra']
    ic_inf = stats_aki['IC Inf']
    ic_sup = stats_aki['IC Sup']
    ancho_ic = ic_sup - ic_inf if not pd.isna(ic_sup) and not pd.isna(ic_inf) else np.nan

    header = f"| {'Estimador Evaluador':<30} | {'Valor Muestra':>13} | {'Población Teórica':>17} | {'IC Inf':>8} | {'IC Sup':>8} | {'Ancho IC':>8} |"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"| {'AKI (Bootstrap Clásico)':<30} | {format_val(aki_val):>13} | {format_val(aki_pob_real):>17} | {format_val(ic_inf):>8} | {format_val(ic_sup):>8} | {format_val(ancho_ic):>8} |")
    print("-" * len(header))
    print(f"⏱️ Tiempo de ejecución: {t1 - t0:.2f} segundos")
    print("=" * len(header) + "\n")

if __name__ == "__main__":
    main()