import argparse
import pandas as pd
import numpy as np
import os
import sys

# Importaciones relativas dentro del paquete
from nmetrics.ordinal.no_core import calcular_estadisticas_no
from nmetrics.ordinal.ako_core import calcular_estadisticas_ako
from nmetrics.ordinal.w_core import calcular_estadisticas_w

def main():
    parser = argparse.ArgumentParser(
        description="📊 ANÁLISIS DE FIABILIDAD GLOBAL: ESCALA ORDINAL (NO, AKO, W)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("archivo", help="Ruta al archivo CSV con la matriz de datos")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, help="Réplicas para SP y Bootstrap (default: 1000)")
    parser.add_argument("-s", "--separador", type=str, default=';', help="Separador del CSV (default: ';')")

    args = parser.parse_args()

    if not os.path.exists(args.archivo):
        print(f"\n❌ Error: No se encontró el archivo '{args.archivo}'.\n")
        sys.exit(1)

    try:
        df = pd.read_csv(args.archivo, sep=args.separador, header=None)
        if df.shape[1] == 1 and args.separador == ';':
            df = pd.read_csv(args.archivo, sep=',', header=None)
        df = df.replace(',', '.', regex=True)
        matriz_empirica = df.values.astype(float)
    except Exception as e:
        print(f"\n❌ Error al procesar el archivo CSV: {e}\n")
        sys.exit(1)

    n_sujetos, m_evaluadores = matriz_empirica.shape

    print("=" * 105)
    print("📊 ANÁLISIS DE FIABILIDAD: ESCALA ORDINAL")
    print("=" * 105)
    print(f" -> Archivo                  : {os.path.basename(args.archivo)}")
    print(f" -> Dimensiones de la matriz : {n_sujetos} sujetos x {m_evaluadores} evaluadores")
    print(f" -> Categorías de la escala  : {args.escala}")
    print(f" -> Réplicas de simulación   : {args.replicas}\n")

    print("-" * 105)
    print(" 📥 MATRIZ DE ENTRADA (Sujetos x Evaluadores):")
    print("-" * 105)
    df_impresion = pd.DataFrame(
        matriz_empirica, 
        index=[f"S{i+1}" for i in range(n_sujetos)],
        columns=[f"E{j+1}" for j in range(m_evaluadores)]
    )
    print(df_impresion.to_string(na_rep="NaN"))
    print("-" * 105 + "\n")

    print(" ⏳ Procesando Ordinal Coefficient (NO) por Simulación Ponderada...")
    no_muestra, no_ponderado, inf_no, sup_no, *_ = calcular_estadisticas_no(matriz_empirica, args.replicas, args.escala, 'SP')

    print(" ⏳ Procesando Krippendorff's Alpha Ordinal (AKO) por Bootstrap Clásico...")
    stats_ako = calcular_estadisticas_ako(matriz_empirica, args.replicas, args.escala)

    print(" ⏳ Procesando Kendall's W por Bootstrap Clásico...")
    stats_w = calcular_estadisticas_w(matriz_empirica, args.replicas, args.escala)

    print("\n" + "=" * 105)
    print(f"| {'Estimador':<35} | {'Valor Muestra':<13} | {'IC Inferior':<12} | {'IC Superior':<12} | {'Ancho IC':<10} |")
    print("-" * 105)
    
    def fmt(v): return f"{v:>13.4f}" if not pd.isna(v) else f"{'N/A':>13}"

    ancho_no = sup_no - inf_no if not (pd.isna(sup_no) or pd.isna(inf_no)) else np.nan
    ancho_ako = stats_ako['IC Sup'] - stats_ako['IC Inf'] if not (pd.isna(stats_ako['IC Sup']) or pd.isna(stats_ako['IC Inf'])) else np.nan
    ancho_w = stats_w['IC Sup'] - stats_w['IC Inf'] if not (pd.isna(stats_w['IC Sup']) or pd.isna(stats_w['IC Inf'])) else np.nan

    print(f"| {'NO (Simulación Ponderada)':<35} | {fmt(no_muestra)} | {fmt(inf_no)} | {fmt(sup_no)} | {fmt(ancho_no):>10} |")
    print(f"| {'Krippendorff Alpha Ordinal (AKO)':<35} | {fmt(stats_ako['AKO Muestra'])} | {fmt(stats_ako['IC Inf'])} | {fmt(stats_ako['IC Sup'])} | {fmt(ancho_ako):>10} |")
    print(f"| {'Kendall W (Bootstrap)':<35} | {fmt(stats_w['W Muestra'])} | {fmt(stats_w['IC Inf'])} | {fmt(stats_w['IC Sup'])} | {fmt(ancho_w):>10} |")
    print("=" * 105)

if __name__ == "__main__":
    main()