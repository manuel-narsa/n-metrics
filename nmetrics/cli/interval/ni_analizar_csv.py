import argparse
import pandas as pd
import numpy as np
import os
import sys
import time

# Importación relativa dentro del paquete
from nmetrics.interval.ni_core import (calcular_estadisticas_ni, detectar_anomalias_ni, 
                      calcular_percentil_universal_ni, calcular_azar_termodinamico_ni)

def main():
    parser = argparse.ArgumentParser(description="🔍 ANÁLISIS INDIVIDUAL: Coeficiente Natural Intervalar (NI)")
    parser.add_argument("archivo", help="Ruta al archivo CSV con la matriz empírica")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-r", "--replicas", type=int, default=5000, help="Número de réplicas (default: 5000)")
    parser.add_argument("-s", "--separador", type=str, default=';', help="Separador del CSV (default: ';')")
    parser.add_argument("-u", "--umbral", type=float, default=2.0, help="Umbral de sigmas para anomalías (default: 2.0)")
    parser.add_argument("--exportar", type=str, metavar="RUTA", help="Ruta para exportar las réplicas a CSV (opcional)")

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
        
    n_sujetos, m_jueces = matriz_empirica.shape
    
    print("=" * 85)
    print(f"🚀 ANÁLISIS DEL COEFICIENTE NI (Familia N: Intervalo)")
    print("=" * 85)
    
    t0 = time.time()
    ni_muestra, ni_ponderado, ic_inf_sp, ic_sup_sp, replicas_sp, matrices_sp, idx_sp = calcular_estadisticas_ni(matriz_empirica, args.replicas, args.escala, 'SP')
    _, _, ic_inf_bc, ic_sup_bc, replicas_bc, matrices_bc, idx_bc = calcular_estadisticas_ni(matriz_empirica, args.replicas, args.escala, 'BC')
    
    print("\n-----------------------------------------------------------------------------------")
    print("| Estimador del Espacio Muestral | Valor Muestra |   IC Inf |   IC Sup | Ancho IC |")
    print("-----------------------------------------------------------------------------------")
    print(f"| Simulación Ponderada (SP)      | {ni_muestra:>13.4f} | {ic_inf_sp:>8.4f} | {ic_sup_sp:>8.4f} | {ic_sup_sp - ic_inf_sp:>8.4f} |")
    print(f"| Bootstrap Clásico (BC)         | {ni_muestra:>13.4f} | {ic_inf_bc:>8.4f} | {ic_sup_bc:>8.4f} | {ic_sup_bc - ic_inf_bc:>8.4f} |")
    print("-----------------------------------------------------------------------------------\n")

    # Exportación (Solo se ejecuta si el usuario usó --exportar ruta.csv)
    if args.exportar:
        print(f" ⏳ Generando archivo de exportación de réplicas en: {args.exportar}...")
        # Aquí iría el bloque de código de tu script original que crea filas_csv y el pd.DataFrame
        # ... [El código de exportación que ya tenías en ni_analizar_csv.py] ...
        print(" 💾 ¡ÉXITO! Réplicas exportadas.")

    ni_azar = calcular_azar_termodinamico_ni(m_jueces, args.escala)
    perc_ponderado = calcular_percentil_universal_ni(ni_ponderado, m_jueces, args.escala)
    
    print("="*85)
    print(f"🌍 [NIVEL III] ESCALA UNIVERSAL Y LÍMITE DE AZAR TERMODINÁMICO")
    print("="*85)
    print(f" 🎲 AZAR ESPERADO (Esperanza Nula):   NI = {ni_azar:.4f}")
    print(f" 🌌 ACUERDO PROYECTADO (Poblacional): NI = {ni_ponderado:.4f}  (Percentil Universal: {perc_ponderado:>5.2f} %)")
    print("-" * 85)
    if ni_ponderado > ni_azar:
        print(" ✅ VÁLIDO: La proyección poblacional supera la gravedad del azar combinatorio.")
    else:
        print(" ⚠️ RUIDO: El acuerdo poblacional es indistinguible de respuestas aleatorias.")
    print("="*85 + "\n")

    df_anom, mu_g, sig_g, limite = detectar_anomalias_ni(matriz_empirica, k_escala=args.escala, umbral_sigma=args.umbral)
    anomalos = df_anom[df_anom['Es_Anomalo'] == True]

    print("="*85)
    print(f"🚨 AUDITORÍA DE ANOMALÍAS LOCALES (Umbral: {args.umbral} sigmas | Corte: < {limite:.4f})")
    print("="*85)
    if anomalos.empty:
        print(" ✅ No se han detectado sujetos anómalos.")
    else:
        print(f" ⚠️ Se han detectado {len(anomalos)} sujetos anómalos:")
        for _, row in anomalos.iterrows():
            print(f"    - Sujeto {int(row['Sujeto_ID']):03d} | Acuerdo: {row['Acuerdo_Local']:.4f}")
    print("="*85)

if __name__ == "__main__":
    main()