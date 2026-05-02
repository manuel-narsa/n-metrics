import argparse
import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# IMPORTACIONES RELATIVAS DEL PAQUETE (¡Adiós importlib y sys.path!)
# ==============================================================================
from nmetrics.interval.ni_core import calcular_estadisticas_ni
from nmetrics.interval.aki_core import calcular_estadisticas_aki
from nmetrics.interval.icc21_core import calcular_estadisticas_icc21

def main():
    # 1. Configurar el lector de comandos de la terminal
    parser = argparse.ArgumentParser(
        description="📊 ANÁLISIS DE FIABILIDAD GLOBAL: ESCALA DE INTERVALO (NI, AKI, ICC)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumento obligatorio: el archivo CSV
    parser.add_argument("archivo", help="Ruta al archivo CSV que contiene la matriz de datos")
    
    # Argumentos opcionales (con valores por defecto)
    parser.add_argument("-k", "--escala", type=int, default=5, 
                        help="Número de categorías de la escala (default: 5)")
    parser.add_argument("-r", "--replicas", type=int, default=1000, 
                        help="Número de réplicas para SP y Bootstrap (default: 1000)")
    parser.add_argument("-s", "--separador", type=str, default=';', 
                        help="Separador del CSV, por ej. ',' o ';' (default: ';')")

    args = parser.parse_args()
    
    ruta_csv = args.archivo
    k_escala = args.escala
    s_replicas = args.replicas
    separador = args.separador

    if not os.path.exists(ruta_csv):
        print(f"\n❌ Error: No se encontró el archivo '{ruta_csv}'.\n")
        sys.exit(1)

    # 2. Leer el CSV de forma robusta
    try:
        df = pd.read_csv(ruta_csv, sep=separador, header=None)
        if df.shape[1] == 1 and separador == ';':
            # Fallback por si el usuario dejó ';' pero el archivo era por comas
            df = pd.read_csv(ruta_csv, sep=',', header=None)
            
        # Reemplazar comas por puntos en los decimales si los hubiera
        df = df.replace(',', '.', regex=True)
        matriz_empirica = df.values.astype(float)
    except Exception as e:
        print(f"\n❌ Error al procesar el archivo CSV: {e}\n")
        sys.exit(1)

    n_sujetos, m_evaluadores = matriz_empirica.shape

    print("=" * 100)
    print("📊 ANÁLISIS DE FIABILIDAD: ESCALA DE INTERVALO")
    print("=" * 100)
    print(f" -> Archivo                  : {os.path.basename(ruta_csv)}")
    print(f" -> Dimensiones de la matriz : {n_sujetos} sujetos x {m_evaluadores} evaluadores")
    print(f" -> Categorías de la escala  : {k_escala}")
    print(f" -> Réplicas de simulación   : {s_replicas}\n")

    print("-" * 100)
    print(" 📥 MATRIZ DE ENTRADA (Sujetos x Evaluadores):")
    print("-" * 100)
    df_impresion = pd.DataFrame(
        matriz_empirica, 
        index=[f"S{i+1}" for i in range(n_sujetos)],
        columns=[f"E{j+1}" for j in range(m_evaluadores)]
    )
    print(df_impresion.to_string(na_rep="NaN"))
    print("-" * 100 + "\n")

    # 3. Ejecutar Motores Matemáticos
    print(" ⏳ Procesando Interval Coefficient (NI) por Simulación Ponderada...")
    ni_muestra, ni_ponderado, inf_ni, sup_ni, *_ = calcular_estadisticas_ni(matriz_empirica, s_replicas, k_escala, 'SP')

    print(" ⏳ Procesando Krippendorff's Alpha (AKI) por Bootstrap Clásico...")
    stats_aki = calcular_estadisticas_aki(matriz_empirica, s_replicas)
    
    print(" ⏳ Procesando Coeficiente de Correlación Intraclase ICC(2,1)...")
    stats_icc = calcular_estadisticas_icc21(matriz_empirica)

    # 4. Formatear y Mostrar Resultados
    print("\n" + "=" * 100)
    print(f"| {'Estimador':<32} | {'Valor Muestra':<13} | {'IC Inferior':<12} | {'IC Superior':<12} | {'Ancho IC':<10} |")
    print("-" * 100)
    
    def fmt(v): return f"{v:>13.4f}" if not pd.isna(v) else f"{'N/A':>13}"

    ancho_ni = sup_ni - inf_ni if not (pd.isna(sup_ni) or pd.isna(inf_ni)) else np.nan
    ancho_aki = stats_aki['IC Sup'] - stats_aki['IC Inf'] if not (pd.isna(stats_aki['IC Sup']) or pd.isna(stats_aki['IC Inf'])) else np.nan
    ancho_icc = stats_icc['IC Sup'] - stats_icc['IC Inf'] if not (pd.isna(stats_icc['IC Sup']) or pd.isna(stats_icc['IC Inf'])) else np.nan

    print(f"| {'NI (Simulación Ponderada)':<32} | {fmt(ni_muestra)} | {fmt(inf_ni)} | {fmt(sup_ni)} | {fmt(ancho_ni):>10} |")
    print(f"| {'Krippendorff Alpha (Bootstrap)':<32} | {fmt(stats_aki['AKI Muestra'])} | {fmt(stats_aki['IC Inf'])} | {fmt(stats_aki['IC Sup'])} | {fmt(ancho_aki):>10} |")
    print(f"| {'ICC(2,1) (F-ANOVA)':<32} | {fmt(stats_icc['ICC Muestra'])} | {fmt(stats_icc['IC Inf'])} | {fmt(stats_icc['IC Sup'])} | {fmt(ancho_icc):>10} |")
    print("=" * 100)

if __name__ == "__main__":
    main()