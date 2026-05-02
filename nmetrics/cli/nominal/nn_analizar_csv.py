import argparse
import pandas as pd
import numpy as np
import os
import sys
import time
import locale

from nmetrics.nominal.nn_core import (calcular_estadisticas_nn, detectar_anomalias_nn, 
                      calcular_percentil_universal_nn, calcular_azar_termodinamico_nn)

def configurar_lectura_regional():
    try:
        locale.setlocale(locale.LC_NUMERIC, '') 
        return locale.localeconv()['decimal_point']
    except:
        return ','

def main():
    parser = argparse.ArgumentParser(description="🔍 ANÁLISIS INDIVIDUAL: Coeficiente Natural Nominal (NN)")
    parser.add_argument("archivo", help="Ruta al archivo CSV con la matriz empírica")
    parser.add_argument("-k", "--escala", type=int, default=5, help="Categorías de la escala (default: 5)")
    parser.add_argument("-r", "--replicas", type=int, default=5000, help="Número de réplicas (default: 5000)")
    parser.add_argument("-s", "--separador", type=str, default=';', help="Separador del CSV (default: ';')")
    parser.add_argument("-u", "--umbral", type=float, default=2.0, help="Umbral de sigmas (default: 2.0)")
    parser.add_argument("--exportar", type=str, metavar="RUTA", help="Ruta para exportar las réplicas a CSV (opcional)")

    args = parser.parse_args()

    if not os.path.exists(args.archivo): 
        print(f"❌ Error: No se encontró el archivo '{args.archivo}'")
        sys.exit(1)
        
    separador_decimal = configurar_lectura_regional()
    mapeo_inverso = None
    
    try:
        df = pd.read_csv(args.archivo, sep=args.separador, decimal=separador_decimal, header=None, skipinitialspace=True)
        if df.shape[1] == 1 and args.separador == ';':
            df = pd.read_csv(args.archivo, sep=',', decimal=separador_decimal, header=None, skipinitialspace=True)
            
        try:
            matriz_empirica = df.values.astype(float)
        except ValueError:
            print(" ℹ️ Detectadas etiquetas nominales de texto. Creando isomorfismo numérico interno...")
            valores_unicos = [v for v in pd.unique(df.values.ravel()) if pd.notna(v) and str(v).strip() != '']
            valores_unicos.sort()
            mapeo_categorias = {val: i+1 for i, val in enumerate(valores_unicos)}
            mapeo_inverso = {i+1: val for i, val in enumerate(valores_unicos)}
            matriz_empirica = df.replace(mapeo_categorias).apply(pd.to_numeric, errors='coerce').values.astype(float)
            
    except Exception as e:
        print(f"❌ Error al leer o convertir los datos: {e}")
        sys.exit(1)
        
    n_sujetos, m_jueces = matriz_empirica.shape
    
    print("=" * 105)
    print(f"🚀 ANÁLISIS DEL COEFICIENTE NN (Familia N: Nominal)")
    print("=" * 105)
    
    t0 = time.time()
    nn_muestra, nn_ponderado, ic_inf_sp, ic_sup_sp, replicas_sp, matrices_sp = calcular_estadisticas_nn(matriz_empirica, args.replicas, args.escala, 'SP')
    _, _, ic_inf_bc, ic_sup_bc, replicas_bc, matrices_bc = calcular_estadisticas_nn(matriz_empirica, args.replicas, args.escala, 'BC')
    
    print(f"\n| Estimador del Espacio Muestral | Valor Muestra |   IC Inf |   IC Sup | Ancho IC |")
    print("-" * 83)
    print(f"| Simulación Ponderada (SP)      | {nn_muestra:>13.4f} | {ic_inf_sp:>8.4f} | {ic_sup_sp:>8.4f} | {ic_sup_sp - ic_inf_sp:>8.4f} |")
    print(f"| Bootstrap Clásico (BC)         | {nn_muestra:>13.4f} | {ic_inf_bc:>8.4f} | {ic_sup_bc:>8.4f} | {ic_sup_bc - ic_inf_bc:>8.4f} |")
    
    if args.exportar:
        print(f"\n ⏳ Generando archivo de exportación de réplicas en: {args.exportar}...")
        max_len = max(len(replicas_sp), len(replicas_bc))
        filas_csv = []
        for r in range(max_len):
            mat_sp = matrices_sp[r] if r < len(matrices_sp) else np.full((n_sujetos, m_jueces), np.nan)
            mat_bc = matrices_bc[r] if r < len(matrices_bc) else np.full((n_sujetos, m_jueces), np.nan)
            val_sp = replicas_sp[r] if r < len(replicas_sp) else np.nan
            val_bc = replicas_bc[r] if r < len(replicas_bc) else np.nan
            
            for i in range(n_sujetos):
                fila_dict = {'Replica_ID': r + 1, 'Sujeto': f"S{i+1:03d}"}
                for j in range(m_jueces):
                    val_juez = mat_sp[i, j]
                    fila_dict[f'E{j+1}_SP'] = (mapeo_inverso[int(val_juez)] if mapeo_inverso else int(val_juez)) if not np.isnan(val_juez) else ""
                fila_dict['NN_SP'] = val_sp if i == 0 else ""
                
                for j in range(m_jueces):
                    val_juez = mat_bc[i, j]
                    fila_dict[f'E{j+1}_BC'] = (mapeo_inverso[int(val_juez)] if mapeo_inverso else int(val_juez)) if not np.isnan(val_juez) else ""
                fila_dict['NN_BC'] = val_bc if i == 0 else ""
                filas_csv.append(fila_dict)
                
        df_replicas = pd.DataFrame(filas_csv)
        df_replicas.to_csv(args.exportar, index=False, sep=';', decimal=separador_decimal)
        print(f" 💾 ¡ÉXITO! Se han exportado {max_len} réplicas matriciales.")
    
    nn_azar = calcular_azar_termodinamico_nn(m_jueces, args.escala)
    p_azar = calcular_percentil_universal_nn(nn_azar, m_jueces, args.escala)
    p_muestra = calcular_percentil_universal_nn(nn_muestra, m_jueces, args.escala)
    p_ponderado = calcular_percentil_universal_nn(nn_ponderado, m_jueces, args.escala)
    
    print("\n" + "="*85)
    print(f"🌍 [NIVEL III] ESCALA UNIVERSAL Y LÍMITE DE AZAR TERMODINÁMICO (NOMINAL)")
    print("="*85)
    print(f" 🎲 AZAR ESPERADO (Esperanza Nula):        NN = {nn_azar:.4f}  (Percentil Universal: {p_azar:>5.2f} %)")
    print(f" 🎯 ACUERDO EMPÍRICO OBTENIDO (Plano):     NN = {nn_muestra:.4f}  (Percentil Universal: {p_muestra:>5.2f} %)")
    print(f" 🌌 ACUERDO PROYECTADO (Poblacional):      NN = {nn_ponderado:.4f}  (Percentil Universal: {p_ponderado:>5.2f} %)")
    print("-" * 85)
    if nn_ponderado > nn_azar: 
        print(" ✅ VÁLIDO: La proyección poblacional supera la gravedad del azar combinatorio.")
    else: 
        print(" ⚠️ RUIDO: El acuerdo poblacional es indistinguible de respuestas aleatorias.")
    
    df_anom, mu_g, sig_g, limite = detectar_anomalias_nn(matriz_empirica, args.escala, args.umbral)
    anomalos = df_anom[df_anom['Es_Anomalo'] == True]

    print("\n" + "="*85)
    print(f"🚨 AUDITORÍA DE ANOMALÍAS LOCALES (Umbral: {args.umbral} sigmas | Corte: < {limite:.4f})")
    print("="*85)
    if anomalos.empty:
        print(" ✅ No se han detectado sujetos anómalos. La matriz es topológicamente estable.")
    else:
        print(f" ⚠️ Se han detectado {len(anomalos)} anomalías:")
        for _, row in anomalos.iterrows():
            print(f"    - Sujeto {int(row['Sujeto_ID']):03d} | Acuerdo: {row['Acuerdo_Local']:.4f}")

if __name__ == "__main__":
    main()