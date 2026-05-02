import argparse
import pandas as pd
import numpy as np
import os
import sys

# Importaciones relativas dentro del paquete
from nmetrics.nominal.nn_core import calcular_estadisticas_nn
from nmetrics.nominal.akn_core import calcular_estadisticas_akn
from nmetrics.nominal.kf_core import calcular_estadisticas_kf

def main():
    parser = argparse.ArgumentParser(
        description="📊 ANÁLISIS DE FIABILIDAD GLOBAL: ESCALA NOMINAL (NN, AKN, KF)",
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
            
        try:
            matriz_empirica = df.replace(',', '.', regex=True).values.astype(float)
        except ValueError:
            print(" ℹ️ Detectadas etiquetas de texto. Creando isomorfismo numérico interno...")
            valores_unicos = [v for v in pd.unique(df.values.ravel()) if pd.notna(v) and str(v).strip() != '']
            valores_unicos.sort()
            mapeo = {val: i+1 for i, val in enumerate(valores_unicos)}
            matriz_empirica = df.replace(mapeo).apply(pd.to_numeric, errors='coerce').values.astype(float)
            
    except Exception as e:
        print(f"\n❌ Error al procesar el archivo CSV: {e}\n")
        sys.exit(1)

    n_sujetos, m_evaluadores = matriz_empirica.shape

    print("=" * 105)
    print("📊 ANÁLISIS DE FIABILIDAD: ESCALA NOMINAL (CATEGÓRICA)")
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

    print(" ⏳ Procesando Nominal Coefficient (NN) por Simulación Ponderada...")
    nn_muestra, nn_ponderado, inf_nn, sup_nn, *_ = calcular_estadisticas_nn(matriz_empirica, args.replicas, args.escala, 'SP')

    print(" ⏳ Procesando Krippendorff's Alpha (AKN) por Bootstrap Clásico...")
    stats_akn = calcular_estadisticas_akn(matriz_empirica, args.replicas, args.escala)

    print(" ⏳ Procesando Fleiss' Kappa (KF) por Bootstrap Clásico...")
    stats_kf = calcular_estadisticas_kf(matriz_empirica, args.replicas, args.escala)

    print("\n" + "=" * 105)
    print(f"| {'Estimador':<35} | {'Valor Muestra':<13} | {'IC Inferior':<12} | {'IC Superior':<12} | {'Ancho IC':<10} |")
    print("-" * 105)
    
    def fmt(v): return f"{v:>13.4f}" if not pd.isna(v) else f"{'N/A':>13}"

    ancho_nn = sup_nn - inf_nn if not (pd.isna(sup_nn) or pd.isna(inf_nn)) else np.nan
    ancho_akn = stats_akn['IC Sup'] - stats_akn['IC Inf'] if not (pd.isna(stats_akn['IC Sup']) or pd.isna(stats_akn['IC Inf'])) else np.nan
    ancho_kf = stats_kf['IC Sup'] - stats_kf['IC Inf'] if not (pd.isna(stats_kf['IC Sup']) or pd.isna(stats_kf['IC Inf'])) else np.nan

    print(f"| {'NN (Simulación Ponderada)':<35} | {fmt(nn_muestra)} | {fmt(inf_nn)} | {fmt(sup_nn)} | {fmt(ancho_nn):>10} |")
    print(f"| {'Krippendorff Alpha Nominal (AKN)':<35} | {fmt(stats_akn['AKN Muestra'])} | {fmt(stats_akn['IC Inf'])} | {fmt(stats_akn['IC Sup'])} | {fmt(ancho_akn):>10} |")
    print(f"| {'Fleiss Kappa (KF)':<35} | {fmt(stats_kf['KF Muestra'])} | {fmt(stats_kf['IC Inf'])} | {fmt(stats_kf['IC Sup'])} | {fmt(ancho_kf):>10} |")
    print("=" * 105)

if __name__ == "__main__":
    main()