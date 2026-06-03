import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

def reset_session_state(hard_reset=True):
    """
    Limpia el estado de la sesión para evitar contaminación de datos
    cuando el usuario cambia de archivo CSV.
    """
    claves_a_mantener = ['config_ui', 'auth_token'] 
    
    for key in list(st.session_state.keys()):
        if key not in claves_a_mantener:
            del st.session_state[key]

def cargar_y_agregar_dataset(file, sep=","):
    """
    Lector de Big Data super optimizado y seguro.
    Lee por trozos (chunks) para no colapsar la RAM y lo convierte 
    directamente al hiperespacio comprimido (Clases de Equivalencia).
    """
    import io
    conteo_estados = Counter()
    total_filas = 0
    
    # 'Rebobinar' el archivo al bit cero para evitar errores de lectura
    file.seek(0)
    
    try:
        reader = pd.read_csv(file, sep=sep, header=None, chunksize=100000, engine='c', low_memory=False)
        for chunk in reader:
            chunk = chunk.dropna(how='all')
            if chunk.empty: continue
            
            # Limpieza y conversión numérica
            chunk = chunk.replace({',': '.'}, regex=True).apply(pd.to_numeric, errors='coerce')
            
            # Compresión Termodinámica: convertimos las filas en tuplas para agruparlas
            filas_como_tuplas = [tuple(x) for x in chunk.values]
            conteo_estados.update(filas_como_tuplas)
            total_filas += len(chunk)
            
        if total_filas == 0: 
            raise ValueError(f"No se encontraron filas válidas con separador '{sep}'.")
            
        return conteo_estados, total_filas
        
    except Exception as e:
        raise Exception(f"Fallo en la lectura: {str(e)}")

def procesar_datos_df(df):
    """
    Convierte un DataFrame (ej. la matriz sintética generada por la app)
    en el formato de diccionario comprimido que el motor N necesita.
    (Esta función sustituye y mejora a la antigua 'procesar_big_data').
    """
    conteo_estados = Counter()
    for _, fila in df.iterrows():
        tuple_fila = tuple(fila.values)
        conteo_estados[tuple_fila] += 1
    return conteo_estados, len(df)

def validar_condiciones_analisis(df_val, topologia, n_total):
    """
    Realiza la auditoría de integridad antes de lanzar los motores matemáticos.
    Comprueba tamaños mínimos y la legalidad de los datos según la topología.
    """
    # 1. Comprobación de existencia
    if df_val is None or (isinstance(df_val, pd.DataFrame) and df_val.empty) or (isinstance(df_val, np.ndarray) and df_val.size == 0):
        return False, "La matriz está vacía o es inválida.", True

    # 2. Comprobación de dimensiones mínimas
    cols = df_val.shape[1] if hasattr(df_val, 'shape') else 0
    if cols < 2:
        return False, "Se necesitan al menos 2 jueces (columnas) para calcular consenso.", True

    # 3. Validación de Decimales (Crítica para topologías discretas)
    if topologia in ["Nominal", "Ordinal"]:
        valores = df_val.values if isinstance(df_val, pd.DataFrame) else df_val
        valores_validos = valores[~np.isnan(valores)]
        
        # Si hay diferencia entre el valor y su redondeo, es que tiene decimales
        if np.any(np.abs(valores_validos - np.round(valores_validos)) > 1e-9):
            return False, "🚨 **Error de Topología:** Las métricas NN y NO exigen categorías discretas enteras. No admiten valores decimales.", True

    # 4. Advertencia de muestra pequeña (No fatal)
    if n_total < 3:
        return False, "Se recomiendan al menos 3 sujetos para un análisis robusto.", False

    return True, "", False