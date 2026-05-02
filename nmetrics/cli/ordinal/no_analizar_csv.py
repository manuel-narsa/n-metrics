import argparse
import pandas as pd
import numpy as np
import os
import sys
import time
import locale

from nmetrics.ordinal.no_core import (
    calcular_estadisticas_no, 
    detectar_anomalias_no, 
    calcular_azar_termodinamico_no_analitico_exacto, 
    calcular_percentil_universal_no_exacto           
)

def configurar_lectura_regional():
    try:
        locale.setlocale(locale.LC_NUMERIC, '') 
        return locale.localeconv()['decimal_point']
    except:
        return ','

def main():
    parser = argparse.ArgumentParser(description="🔍 ANÁLISIS INDIVIDUAL: Coeficiente Natural Ordinal (NO)")
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
        
    separador_decimal = configurar_lectura_regional()
    
    try:
        df = pd.read_csv(args.archivo, header=None, sep=args.separador, decimal=separador_decimal, skipinitialspace=True)
        if df.shape[1] == 1 and args.separador == ';':
            df = pd.read_csv(args.archivo, header=None, sep=',', decimal=separador_decimal, skipinitialspace=True)
        matriz_empirica = df.values.astype(float)
    except Exception as e:
        print(f"❌ Error al leer o convertir los datos a números: {e}")
        sys.exit(1)
        
    n_sujetos, m_jueces = matriz_empirica.shape
    
    print("=" * 105)
    print(f"🚀 ANÁLISIS DEL COEFICIENTE NO (Familia N: Ordinal)")
    print("=" * 105)
    
    t0 = time.time()
    no_muestra, no_ponderado, ic_inf_sp, ic_sup_sp, replicas_sp, matrices_sp, idx_sp = calcular_estadisticas_no(matriz_empirica, args.replicas, args.escala, 'SP')
    _, _, ic_inf_bc, ic_sup_bc, replicas_bc, matrices_bc, idx_bc = calcular_estadisticas_no(matriz_empirica, args.replicas, args.escala, 'BC')
    t1 = time.time()

    print("\n-----------------------------------------------------------------------------------")
    print("| Estimador del Espacio Muestral | Valor Muestra |   IC Inf |   IC Sup | Ancho IC |")
    print("-----------------------------------------------------------------------------------")
    print(f"| Simulación Ponderada (SP)      | {no_muestra:>13.4f} | {ic_inf_sp:>8.4f} | {ic_sup_sp:>8.4f} | {ic_sup_sp - ic_inf_sp:>8.4f} |")
    print(f"| Bootstrap Clásico (BC)         | {no_muestra:>13.4f} | {ic_inf_bc:>8.4f} | {ic_sup_bc:>8.4f} | {ic_sup_bc - ic_inf_bc:>8.4f} |")
    print("-----------------------------------------------------------------------------------")
    print(f" ⏱️ Tiempo de procesamiento de simulaciones: {t1-t0:.3f} segundos\n")

    if args.exportar:
        print(f" ⏳ Generando archivo de exportación de réplicas en: {args.exportar}...")
        filas_csv = []
        for i in range(n_sujetos):
            fila_dict = {'Replica_ID': 1, 'Sujeto_SP': f"S{i+1:03d}"}
            for j in range(m_jueces):
                val = matriz_empirica[i, j]
                fila_dict[f'E{j+1}_SP'] = int(val) if not np.isnan(val) else ""
            fila_dict['NO_SP'] = no_ponderado if i == 0 else ""
            fila_dict[' || '] = " | "
            fila_dict['Sujeto_BC'] = f"S{i+1:03d}"
            for j in range(m_jueces):
                val = matriz_empirica[i, j]
                fila_dict[f'E{j+1}_BC'] = int(val) if not np.isnan(val) else ""
            fila_dict['NO_BC'] = no_muestra if i == 0 else ""
            filas_csv.append(fila_dict)

        max_len = max(len(replicas_sp), len(replicas_bc))
        for r in range(max_len):
            mat_sp = matrices_sp[r] if r < len(matrices_sp) else np.full((n_sujetos, m_jueces), np.nan)
            mat_bc = matrices_bc[r] if r < len(matrices_bc) else np.full((n_sujetos, m_jueces), np.nan)
            val_sp = replicas_sp[r] if r < len(replicas_sp) else np.nan
            val_bc = replicas_bc[r] if r < len(replicas_bc) else np.nan
            idx_orig_sp = idx_sp[r] if r < len(idx_sp) else np.zeros(n_sujetos, dtype=int)
            idx_orig_bc = idx_bc[r] if r < len(idx_bc) else np.zeros(n_sujetos, dtype=int)
            
            for i in range(n_sujetos):
                fila_dict = {'Replica_ID': r + 2}
                fila_dict['Sujeto_SP'] = f"S{idx_orig_sp[i]+1:03d}"
                for j in range(m_jueces):
                    val = mat_sp[i, j]
                    fila_dict[f'E{j+1}_SP'] = int(val) if not np.isnan(val) else ""
                fila_dict['NO_SP'] = val_sp if i == 0 else ""
                
                fila_dict[' || '] = " | "
                
                fila_dict['Sujeto_BC'] = f"S{idx_orig_bc[i]+1:03d}"
                for j in range(m_jueces):
                    val = mat_bc[i, j]
                    fila_dict[f'E{j+1}_BC'] = int(val) if not np.isnan(val) else ""
                fila_dict['NO_BC'] = val_bc if i == 0 else ""
                filas_csv.append(fila_dict)
                
        df_replicas = pd.DataFrame(filas_csv)
        df_replicas.to_csv(args.exportar, index=False, sep=';', decimal=separador_decimal)
        print(f" 💾 ¡ÉXITO! Se han exportado {max_len} réplicas matriciales.\n")

    no_azar, _, _ = calcular_azar_termodinamico_no_analitico_exacto(n_sujetos, m_jueces, args.escala)
    p_azar, _ = calcular_percentil_universal_no_exacto(no_azar, n_sujetos, m_jueces, args.escala)
    p_muestra, _ = calcular_percentil_universal_no_exacto(no_muestra, n_sujetos, m_jueces, args.escala)
    p_ponderado, _ = calcular_percentil_universal_no_exacto(no_ponderado, n_sujetos, m_jueces, args.escala)
    
    print("="*85)
    print(f"🌍 [NIVEL III] ESCALA UNIVERSAL Y LÍMITE DE AZAR TERMODINÁMICO (ORDINAL)")
    print("="*85)
    print(f" 🎲 AZAR ESPERADO (Esperanza Nula):        NO = {no_azar:.4f}  (Percentil: {p_azar:>5.2f} %)")
    print(f" 🎯 ACUERDO EMPÍRICO OBTENIDO (Plano):     NO = {no_muestra:.4f}  (Percentil: {p_muestra:>5.2f} %)")
    print(f" 🌌 ACUERDO PROYECTADO (Poblacional):      NO = {no_ponderado:.4f}  (Percentil: {p_ponderado:>5.2f} %)")
    print("-" * 85)
    if no_ponderado > no_azar: 
        print(" ✅ VÁLIDO: La proyección poblacional supera la gravedad del azar combinatorio.")
    else: 
        print(" ⚠️ RUIDO: El acuerdo poblacional es indistinguible de respuestas aleatorias.")
    print("="*85 + "\n")

    df_anom, mu_g, sig_g, limite = detectar_anomalias_no(matriz_empirica, k_escala=args.escala, umbral_sigma=args.umbral)
    
    if not df_anom.empty:
        anomalos = df_anom[df_anom['Es_Anomalo'] == True]
        print("="*80)
        print(f"🚨 AUDITORÍA DE ANOMALÍAS DE EVALUADORES (Umbral: {args.umbral} sigmas | Corte: < {limite:.4f})")
        print("="*80)
        if anomalos.empty:
            print(" ✅ No se han detectado Jueces anómalos. El panel es topológicamente estable.")
        else:
            print(f" ⚠️ ¡ATENCIÓN! Se han detectado {len(anomalos)} Jueces que rompen el consenso ordinal:")
            for _, row in anomalos.iterrows():
                print(f"    - Juez {int(row['Juez_ID']):03d} | Acuerdo Local: {row['Acuerdo_Local']:.4f}")
        print("="*80)

if __name__ == "__main__":
    main()