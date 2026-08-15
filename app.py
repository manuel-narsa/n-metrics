import os
import sys
import json
import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from collections import Counter

# ==============================================================================
# FIJAR RUTA PRINCIPAL 
# ==============================================================================
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# IMPORTACIONES DEL MOTOR TERMODINÁMICO (N-Metrics)
# ==============================================================================
from nmetrics.interval import ni_core, aki_core, icc21_core
from nmetrics.nominal import nn_core, akn_core, kf_core
from nmetrics.ordinal import no_core, ako_core, w_core

from nmetrics.utils.data_handler import reset_session_state, cargar_y_agregar_dataset, validar_condiciones_analisis
from nmetrics.utils.simulations import ejecutar_auditoria_cobertura, ejecutar_duelo_ia

DB_DICCIONARIOS_PATH = "diccionarios_estados.json"

# ==============================================================================
# INICIALIZACIÓN DEL SESSION STATE
# ==============================================================================
if "base_diccionarios" not in st.session_state:
    if os.path.exists(DB_DICCIONARIOS_PATH):
        try:
            with open(DB_DICCIONARIOS_PATH, "r", encoding="utf-8") as f:
                st.session_state["base_diccionarios"] = json.load(f)
        except Exception:
            st.session_state["base_diccionarios"] = {}
    else:
        st.session_state["base_diccionarios"] = {}

if "diccionario_estados" not in st.session_state:
    st.session_state["diccionario_estados"] = {}

# ==============================================================================
# GESTIÓN DE ARCHIVOS Y BASE DE DATOS DE FIRMAS (JSON PRECALCULADO)
# ==============================================================================
@st.cache_data(show_spinner=False)
def cargar_firmas_precalculadas(m: int):
    """
    Carga el diccionario precalculado de firmas/clases de equivalencia 
    para una dimensión m específica desde la carpeta 'data/'.
    """
    path_json = os.path.join("data", f"firmas_m_{m}.json")
    if os.path.exists(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def calcular_firma(vector):
    """Calcula la firma de un vector de estado de forma secuencial (Fallback)."""
    conteo = Counter(vector)
    return tuple(sorted(conteo.values(), reverse=True))

def cargar_base_diccionarios(filepath=DB_DICCIONARIOS_PATH):
    """Carga la base de datos de diccionarios de macroestados desde disco (JSON)."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
    
def sanitizar_para_json(obj):
    """Convierte recursivamente las claves a cadenas limpias para JSON."""
    if isinstance(obj, dict):
        dict_limpio = {}
        for k, v in obj.items():
            if isinstance(k, (tuple, list)):
                clave_limpia = str(tuple(int(x) for x in k))
            else:
                clave_limpia = str(k)
            dict_limpio[clave_limpia] = sanitizar_para_json(v)
        return dict_limpio
    elif isinstance(obj, list):
        return [sanitizar_para_json(elem) for elem in obj]
    return obj
    
def guardar_base_diccionarios(base_dict, filepath=DB_DICCIONARIOS_PATH):
    """Persiste la base de datos de diccionarios en un archivo JSON local."""
    try:
        dict_preparado = sanitizar_para_json(base_dict)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dict_preparado, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

# ==============================================================================
# MOTOR DE INFERENCIA DE UI 
# ==============================================================================

def ejecutar_calculo_optimizado(dict_estados, topologia, k_calc, replicas, estimadores, m_jueces, n_total, matriz_np):
    """Enrutador principal de cálculos que alimenta las tablas de la interfaz."""
    resultados_inf = []
    
    # --- A. MOTOR UNIFICADO: NI (Intervalar) ---
    if "Intervalar" in topologia and "NI (Marco N)" in estimadores:
        try:
            with st.status("Calculando Coeficiente NI...", expanded=True) as status:
                st.write("Procesando matriz y masa entrópica en la nube...")
                ni_muestra, pob_real, inf, sup = ni_core.calcular_estadisticas_ni_unificada(dict_estados, k_calc, replicas)
                p_e = ni_core.calcular_azar_termodinamico_ni(m_jueces, k_calc)
                perc = ni_core.calcular_percentil_universal_ni(pob_real, m_jueces, k_calc)
                status.update(label="¡Cálculo NI completado con éxito!", state="complete", expanded=False)

            resultados_inf.append({
                "Métrica": "NI", "Muestra": ni_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": p_e, "Percentil (%)": perc, "Motor": "Simulación Ponderada"
            })
        except Exception as e: 
            st.warning(f"⚠️ El motor NI falló: {e}")

    # --- B. MOTOR UNIFICADO: NN (Nominal) ---
    elif "Nominal" in topologia and "NN (Marco N)" in estimadores:
        try:
            nn_muestra, pob_real, inf, sup = nn_core.calcular_estadisticas_nn_unificada(dict_estados, k_calc, replicas)
            p_e = nn_core.calcular_azar_termodinamico_nn(m_jueces, k_calc)
            perc = nn_core.calcular_percentil_universal_nn(pob_real, m_jueces, k_calc)
            resultados_inf.append({
                "Métrica": "NN", "Muestra": nn_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": p_e, "Percentil (%)": perc, "Motor": "Simulación Ponderada"
            })
        except Exception as e: st.warning(f"⚠️ El motor NN falló: {e}")

    # --- C. MOTOR UNIFICADO: NO (Ordinal) ---
    elif "Ordinal" in topologia and "NO (Marco N)" in estimadores:
        try:
            no_muestra, pob_real, inf, sup = no_core.calcular_estadisticas_no_unificada(dict_estados, k_calc, replicas)
            p_e, _, perc, min_N = no_core.analizar_termodinamica_no(n_total, m_jueces, k_calc, valor_observado=pob_real)
            
            resultados_inf.append({
                "Métrica": "NO", "Muestra": no_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": p_e, "Percentil (%)": perc, "Motor": "Simulación Ponderada"
            })
        except Exception as e: st.warning(f"⚠️ El motor NO falló: {e}")

    # --- D. ESTIMADORES CLÁSICOS ---
    if n_total <= 50000:
        if "Intervalar" in topologia:
            if "AKI (Bootstrap C.)" in estimadores:
                try:
                    p_aki = aki_core.calcular_aki_poblacion_asintotica(matriz_np, 1, k_calc, 100)
                    st_aki = aki_core.calcular_estadisticas_aki(matriz_np, replicas)
                    resultados_inf.append({
                        "Métrica": "AKI", "Muestra": st_aki['AKI Muestra'], "Pob. Real": p_aki, 
                        "IC Inf": st_aki['IC Inf'], "IC Sup": st_aki['IC Sup'], 
                        "Ancho IC": st_aki['IC Sup'] - st_aki['IC Inf'], "Motor": "Bootstrap Clásico"
                    })
                except Exception as e: st.warning(f"Error en AKI: {e}")
            if "ICC(2,1) (F-ANOVA)" in estimadores:
                try:
                    p_icc = icc21_core.calcular_icc_poblacion_asintotica(matriz_np, 1, k_calc, 100)
                    st_icc = icc21_core.calcular_estadisticas_icc21(matriz_np)
                    resultados_inf.append({
                        "Métrica": "ICC(2,1)", "Muestra": st_icc['ICC Muestra'], "Pob. Real": p_icc, 
                        "IC Inf": st_icc['IC Inf'], "IC Sup": st_icc['IC Sup'], 
                        "Ancho IC": st_icc['IC Sup'] - st_icc['IC Inf'], "Motor": "F-ANOVA"
                    })
                except Exception as e: st.warning(f"Error en ICC(2,1): {e}")

        if "Nominal" in topologia:
            if "AKN (Bootstrap C.)" in estimadores:
                try:
                    p_akn = akn_core.calcular_akn_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_akn = akn_core.calcular_estadisticas_akn(matriz_np, replicas)
                    resultados_inf.append({
                        "Métrica": "AKN", "Muestra": st_akn['AKN Muestra'], "Pob. Real": p_akn, 
                        "IC Inf": st_akn['IC Inf'], "IC Sup": st_akn['IC Sup'], 
                        "Ancho IC": st_akn['IC Sup'] - st_akn['IC Inf'], "Motor": "Bootstrap Clásico"
                    })
                except Exception as e: st.warning(f"Error en AKN: {e}")
            if "Kappa Fleiss (Bootstrap C.)" in estimadores:
                try:
                    p_kf = kf_core.calcular_kf_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_kf = kf_core.calcular_estadisticas_kf(matriz_np, replicas)
                    resultados_inf.append({
                        "Métrica": "KF", "Muestra": st_kf['KF Muestra'], "Pob. Real": p_kf, 
                        "IC Inf": st_kf['IC Inf'], "IC Sup": st_kf['IC Sup'], 
                        "Ancho IC": st_kf['IC Sup'] - st_kf['IC Inf'], "Motor": "Varianza asintótica (Fleiss, 2003)"
                    })
                except Exception as e: st.warning(f"Error en KF: {e}")

        if "Ordinal" in topologia:
            if "AKO (Bootstrap C.)" in estimadores:
                try:
                    p_ako = ako_core.calcular_ako_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_ako = ako_core.calcular_estadisticas_ako(matriz_np, replicas)
                    resultados_inf.append({
                        "Métrica": "AKO", "Muestra": st_ako['AKO Muestra'], "Pob. Real": p_ako, 
                        "IC Inf": st_ako['IC Inf'], "IC Sup": st_ako['IC Sup'], 
                        "Ancho IC": st_ako['IC Sup'] - st_ako['IC Inf'], "Motor": "Bootstrap Clásico"
                    })
                except Exception as e: st.warning(f"Error en AKO: {e}")
            if "Kendall W (Bootstrap C.)" in estimadores:
                try:
                    p_w = w_core.calcular_w_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_w = w_core.calcular_estadisticas_w(matriz_np, replicas)
                    resultados_inf.append({
                        "Métrica": "W", "Muestra": st_w['W Muestra'], "Pob. Real": p_w, 
                        "IC Inf": st_w['IC Inf'], "IC Sup": st_w['IC Sup'], 
                        "Ancho IC": st_w['IC Sup'] - st_w['IC Inf'], "Motor": "Bootstrap Clásico"
                    })
                except Exception as e: st.warning(f"Error en W: {e}")

    return resultados_inf

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_icono = os.path.join(BASE_DIR, "icono.png")

try:
    if os.path.exists(ruta_icono):
        img_icono = Image.open(ruta_icono)
        st.set_page_config(page_title="Métricas N: Termodinámica del Consenso", page_icon=img_icono, layout="wide")
    else:
        st.set_page_config(page_title="Métricas N: Termodinámica del Consenso", layout="wide")
except Exception:
    pass

st.markdown("""
    <style>
        .block-container { padding-bottom: 10rem !important; }
    </style>
""", unsafe_allow_html=True)

if "topologia_activa" not in st.session_state: st.session_state["topologia_activa"] = "Intervalar (Discreta/Continua)"
if "pestaña_activa" not in st.session_state: st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
topologia = st.session_state["topologia_activa"].split(" ")[0]

# ==============================================================================
# PANEL LATERAL (SIDEBAR)
# ==============================================================================
st.sidebar.header("📂 Carga de Datos")
separador = st.sidebar.selectbox("Separador del CSV", [",", ";", "\t"], key="sep_main", on_change=reset_session_state)
archivo_subido = st.sidebar.file_uploader("Sube tu matriz empírica (CSV, TXT)", type=["csv", "txt"], on_change=reset_session_state)

if archivo_subido is not None and ('df_original_raw' not in st.session_state or st.session_state['df_original_raw'] is None):
    with st.sidebar.status("⚙️ Procesando matriz...", expanded=True) as status:
        try:
            dict_est, n_total = cargar_y_agregar_dataset(archivo_subido, separador)
            st.session_state.diccionario_estados = dict_est
            st.session_state.n_total = n_total
            st.session_state["topologia_activa"] = "Intervalar (Discreta/Continua)"
            st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
            
            if "base_diccionarios" not in st.session_state:
                st.session_state["base_diccionarios"] = {}

            nombre_ds = getattr(archivo_subido, "name", "Dataset_Cargado")
            st.session_state["base_diccionarios"][nombre_ds] = dict_est
            guardar_base_diccionarios(st.session_state["base_diccionarios"])
            
            archivo_subido.seek(0)
            df_raw_loaded = pd.read_csv(archivo_subido, sep=separador, header=None)
            df_raw_loaded = df_raw_loaded.replace({',': '.'}, regex=True).apply(pd.to_numeric, errors='coerce')
            st.session_state.df_original_raw = df_raw_loaded
            
            status.update(label="✅ Matriz procesada e integrada en BD.", state="complete")
            st.rerun() 
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")

if 'df_original_raw' in st.session_state and st.session_state['df_original_raw'] is not None and len(st.session_state['df_original_raw']) > 0:
    st.sidebar.success(f"✅ Matriz activa: {st.session_state.get('n_total', 0):,} sujetos.")
    if st.session_state.get('usar_sintetica'):
        if st.sidebar.button("🗑️ Borrar Matriz Sintética", use_container_width=True):
            reset_session_state()
            st.session_state["topologia_activa"] = "Intervalar (Discreta/Continua)"
            st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Diccionarios en BD")

base_bd = st.session_state.get("base_diccionarios", {})

if base_bd:
    opciones_dict = ["-- Seleccionar Dataset Guardado --"] + list(base_bd.keys())
    seleccion_dict = st.sidebar.selectbox("Diccionarios guardados", opciones_dict, key="sel_dict_bd")
    
    if seleccion_dict != "-- Seleccionar Dataset Guardado --":
        if st.sidebar.button("📥 Cargar Diccionario Seleccionado", use_container_width=True):
            st.session_state["diccionario_estados"] = base_bd[seleccion_dict]
            st.sidebar.success(f"Diccionario '{seleccion_dict}' activo en memoria.")
            st.rerun()
else:
    st.sidebar.caption("No hay diccionarios guardados en disco.")

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Configuración de Escala")
v_min_user = st.sidebar.number_input("Valor MÍNIMO", value=1.0)
v_max_user = st.sidebar.number_input("Valor MÁXIMO", value=5.0)
replicas = st.sidebar.slider("Réplicas de Simulación (S)", 100, 5000, 1000)

topologia_opciones = ["Intervalar (Discreta/Continua)", "Nominal (Categórica)", "Ordinal (Ordenada)"]
indice_actual = topologia_opciones.index(st.session_state["topologia_activa"]) if st.session_state.get("topologia_activa") in topologia_opciones else 0
seleccion_user = st.sidebar.radio("Naturaleza de los datos:", topologia_opciones, index=indice_actual)

if seleccion_user != st.session_state.get("topologia_activa"):
    st.session_state["topologia_activa"] = seleccion_user
    st.session_state.pop('res_inferencia', None)
    st.rerun()

# ==============================================================================
# PROCESAMIENTO TOPOLÓGICO Y VARIABLES GLOBALES
# ==============================================================================
matriz_original, matriz_empirica = None, None  
k_escala = None
n_sujetos, m_jueces = 0, 0
escala_valida = True  

limite_inferior = float(v_min_user) 
limite_superior = float(v_max_user)

if st.session_state.get('usar_sintetica') and st.session_state.get('matriz_generada_app') is not None:
    df_gen = st.session_state['matriz_generada_app']
    n_sujetos, m_jueces = df_gen.shape
    matriz_original = df_gen.values

elif 'df_original_raw' in st.session_state and not st.session_state.get('usar_sintetica'):
    matriz_original = st.session_state['df_original_raw'].values
    n_sujetos, m_jueces = matriz_original.shape

if matriz_original is not None:
    try:
        if not np.isnan(matriz_original).all():
            val_max = float(np.nanmax(matriz_original))
            val_min = float(np.nanmin(matriz_original))
            
            if val_max > (limite_superior + 0.00001) or val_min < (limite_inferior - 0.00001):
                st.sidebar.error(f"🚨 **Escala Inconsistente:** La matriz contiene valores (de {val_min} a {val_max}) que DESBORDAN los límites teóricos declarados [{limite_inferior}, {limite_superior}].")
                escala_valida = False

        if escala_valida:
            rango_escala = limite_superior - limite_inferior
            if rango_escala <= 0:
                st.sidebar.error("🚨 El límite superior debe ser mayor que el límite inferior.")
                escala_valida = False
            else:
                k_escala = int(round(rango_escala + 1)) 
                matriz_empirica = 1 + (matriz_original - limite_inferior) * ((k_escala - 1) / rango_escala)
                matriz_empirica = np.clip(matriz_empirica, 1.0, float(k_escala))
                
                st.session_state['matriz_empirica'] = matriz_empirica
                st.session_state['k_escala'] = k_escala
                
    except Exception as e: 
        st.sidebar.error(f"Error en el procesamiento matemático: {e}")
        escala_valida = False

if not escala_valida:
    st.session_state.pop('matriz_empirica', None)
    st.session_state.pop('res_inferencia', None)
    st.warning("⚠️ **El cálculo está bloqueado.** Has introducido unos límites de escala incompatibles con la matriz actual. Por favor, corrige los valores en el menú lateral.")
    st.stop()

# ==============================================================================
# CABECERA Y NAVEGACIÓN
# ==============================================================================
col_logo, col_titulo = st.columns([1, 15], vertical_alignment="center") 
with col_logo:
    try: st.image("icono.png", width=60)
    except Exception: pass
with col_titulo:
    st.markdown('<h1 style="margin-top: 0rem; padding-top: 0rem;">Métricas N: La Termodinámica del Consenso</h1>', unsafe_allow_html=True)
st.markdown("Plataforma oficial para la inferencia termodinámica, auditoría topológica y procesamiento a escala de Big Data de matrices empíricas.")

opciones_pestañas = [
    "📊 Cálculo del Consenso", 
    "🎯 Pruebas de Cobertura", 
    "📈 Informe Multirrango", 
    "🔄 Invarianza (n) y Sensibilidad (m)", 
    "🏗️ Generador de Matrices", 
    "⚔️ Duelo: N vs Clásicos", 
    "📚 BD Diccionarios",
    "📖 Manual de Usuario", 
    "📜 Autoría y Licencia"
]
pestaña_seleccionada = st.radio("Navegación del sistema:", options=opciones_pestañas, horizontal=True, index=opciones_pestañas.index(st.session_state["pestaña_activa"]), label_visibility="collapsed")

if pestaña_seleccionada != st.session_state["pestaña_activa"]:
    st.session_state["pestaña_activa"] = pestaña_seleccionada
    st.rerun()
st.markdown("---")

# ==============================================================================
# PESTAÑA 1: INFERENCIA DE CONSENSO Y ANOMALÍAS
# ==============================================================================
if st.session_state["pestaña_activa"] == '📊 Cálculo del Consenso':
    if matriz_original is not None:
        st.success(f"🚀 Análisis Activo: Procesando {n_sujetos:,} registros con {m_jueces} jueces y escala {k_escala}.")
        with st.expander("👁️ Ver Matriz", expanded=False):
            df_vis = pd.DataFrame(matriz_original, columns=[f"J{j+1:03d}" for j in range(m_jueces)], index=[f"S{i+1:03d}" for i in range(n_sujetos)])
            st.dataframe(df_vis, use_container_width=True)

    if "Intervalar" in topologia: opciones = ["NI (Marco N)", "AKI (Bootstrap C.)", "ICC(2,1) (F-ANOVA)"]
    elif "Nominal" in topologia: opciones = ["NN (Marco N)", "AKN (Bootstrap C.)", "Kappa Fleiss (Bootstrap C.)"]
    else: opciones = ["NO (Marco N)", "AKO (Bootstrap C.)", "Kendall W (Bootstrap C.)"]
        
    estimadores = st.multiselect("Selecciona modelos:", opciones, default=[opciones[0]])

    if st.button("🚀 Calcular Coeficientes", type="primary"):
        t_inicio = time.time()
        dict_ce = st.session_state.get("diccionario_estados")
        
        if dict_ce is None: st.error("🚨 La matriz no está cargada."); st.stop()
        valido, msg, fatal = validar_condiciones_analisis(matriz_original, topologia, n_sujetos)
        if fatal: st.error(msg); st.stop()
        elif msg: st.warning(msg)

        with st.spinner("Procesando..."):
            resultados = ejecutar_calculo_optimizado(dict_ce, topologia, k_escala, replicas, estimadores, m_jueces, n_sujetos, matriz_original)
            st.session_state['res_inferencia'] = {'df': pd.DataFrame(resultados), 'tiempo': time.time() - t_inicio}
            st.rerun()

    if st.session_state.get('res_inferencia'):
        res = st.session_state['res_inferencia']
        df_res = res['df']
        
        if not df_res.empty:
            st.success(f"⏱️ Tiempo total de ejecución: {res['tiempo']:.4f} segundos.")
            columnas_posibles = ['Métrica', 'Muestra', 'Pob. Real', 'IC Inf', 'IC Sup', 'Ancho IC', 'Valor Azar', 'Percentil (%)', 'Motor']
            columnas_a_mostrar = [c for c in columnas_posibles if c in df_res.columns]
            
            def format_4d(x): return f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
            def format_2d(x): return f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
            formatos = {col: format_4d for col in ['Muestra', 'Pob. Real', 'IC Inf', 'IC Sup', 'Ancho IC', 'Valor Azar'] if col in columnas_a_mostrar}
            if "Percentil (%)" in columnas_a_mostrar: formatos["Percentil (%)"] = format_2d

            def auditar(row):
                estilos = [''] * len(row)
                try:
                    if 'IC Inf' in row and 'IC Sup' in row:
                        inf, sup = float(row['IC Inf']), float(row['IC Sup'])
                        for col in ['Muestra', 'Pob. Real']:
                            if col in row and pd.notna(row[col]):
                                val = float(row[col])
                                idx = list(row.index).index(col)
                                estilos[idx] = 'color: #28a745; font-weight: bold;' if inf <= val <= sup else 'color: #dc3545; font-weight: bold;'
                except Exception: pass
                return estilos

            st.dataframe(df_res[columnas_a_mostrar].style.format(formatos).apply(auditar, axis=1), use_container_width=True)
            
        st.markdown("---")
        st.markdown(f"### 🔍 Escáner de Anomalías ({'Jueces Disidentes' if 'Ordinal' in topologia else 'Macroestados'})")
        umbral_sigma = st.slider("Umbral de tolerancia ($\\sigma$):", 0.5, 3.0, 1.5, 0.1)

        if st.button("🔎 Ejecutar Escáner", type="secondary"):
            with st.spinner("Buscando perturbaciones termodinámicas..."):
                try:
                    if "Intervalar" in topologia:
                        matriz_escaner, k_escaner = matriz_empirica, k_escala
                    else:
                        valores_unicos = np.unique(matriz_original[~np.isnan(matriz_original)])
                        k_escaner = len(valores_unicos)
                        mapeo = {val: i+1 for i, val in enumerate(sorted(valores_unicos))}
                        matriz_escaner = np.full_like(matriz_original, np.nan, dtype=float)
                        for old_val, new_val in mapeo.items(): matriz_escaner[matriz_original == old_val] = new_val

                    if "Intervalar" in topologia:
                        df_anom, mu_g, sig_g, lim = ni_core.detectar_anomalias_ni(matriz_escaner, k_escaner, umbral_sigma)
                        tipo = "Sujetos"
                    elif "Nominal" in topologia:
                        df_anom, mu_g, sig_g, lim = nn_core.detectar_anomalias_nn(matriz_escaner, k_escaner, umbral_sigma)
                        tipo = "Sujetos"
                    else:
                        df_anom, mu_g, sig_g, lim = no_core.detectar_anomalias_no(matriz_escaner, k_escaner, umbral_sigma)
                        tipo = "Jueces"

                    if not df_anom.empty:
                        prefix = "S" if tipo == "Sujetos" else "J"
                        col_id = "Sujeto_ID" if tipo == "Sujetos" else "Juez_ID"
                        df_anom['ID'] = [f"{prefix}{int(i):03d}" for i in df_anom[col_id]]
                        df_anom.set_index('ID', inplace=True)
                        df_anom.drop(columns=[col_id], inplace=True)

                    st.session_state['res_anomalias'] = {"df": df_anom, "mu": mu_g, "sig": sig_g, "limite": lim, "tipo": tipo}
                    st.toast("🛡️ Escaneo completado.", icon="✅")
                except Exception as e: st.error(f"⚠️ Error: {e}")

        if st.session_state.get('res_anomalias'):
            datos_a = st.session_state['res_anomalias']
            if pd.isna(datos_a['mu']): st.warning("Datos insuficientes para buscar anomalías.")
            else:
                st.write(f"Acuerdo Medio Global: **{datos_a['mu']:.4f}** (Límite crítico: < {datos_a['limite']:.4f})")
                df_completo = datos_a['df']
                if not df_completo.empty:
                    df_anomalos = df_completo[df_completo['Es_Anomalo'] == True].copy()
                    if not df_anomalos.empty:
                        st.warning(f"🚨 Detectados {len(df_anomalos)} {datos_a['tipo'].lower()} anómalos:")
                        df_visual = df_anomalos[['Acuerdo_Local']].rename(columns={'Acuerdo_Local': 'Acuerdo Local (A)'})
                        st.dataframe(df_visual.style.format("{:.4f}").apply(lambda x: ['background-color: #fee2e2; color: #dc3545; font-weight: bold;']*len(x), axis=1), use_container_width=True)
                    else: st.success(f"✨ Sistema estable bajo $\\sigma = {umbral_sigma}$.")

# ==============================================================================
# PESTAÑA 2: PRUEBAS DE COBERTURA
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🎯 Pruebas de Cobertura':
    st.markdown("### Auditoría de Cobertura Termodinámica")
    if matriz_original is None:
        st.warning("⚠️ **Aviso:** Debes cargar un archivo de datos (.csv o .txt) desde el panel lateral izquierdo antes de usar esta sección.")
    elif k_escala is None:
        st.warning("⚠️ **Aviso:** Debes verificar los límites de la escala (k) en el panel lateral.")
    else:
        dim_n, dim_m = n_sujetos if matriz_empirica is not None else 50, m_jueces if matriz_empirica is not None else 7
        if matriz_empirica is not None and dim_n > 1500:
            dim_n = 1500

        col_dim1, col_dim2 = st.columns(2)
        if matriz_empirica is None:
            with col_dim1: dim_n = st.number_input("Sujetos (n)", 5, 2000, 50, 5)
            with col_dim2: dim_m = st.number_input("Jueces (m)", 2, 30, 7, 1)
        
        with st.spinner("Calculando límites del hiperespacio..."):
            if "Intervalar" in topologia: 
                azar_N = ni_core.calcular_azar_termodinamico_ni(dim_m, k_escala)
                min_N = 0.0 
                opciones_cob = ["NI (Marco N)", "Alpha Krippendorff", "ICC(2,1)"]
            elif "Nominal" in topologia:
                azar_N = nn_core.calcular_azar_termodinamico_nn(dim_m, k_escala)
                min_N = np.sqrt(min(nn_core._build_macrostate_dictionary_nn(dim_m, k_escala).keys())) 
                opciones_cob = ["NN (Marco N)", "Alpha Krippendorff", "Kappa Fleiss"]
            else: 
                azar_N, _, _, min_N = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala)
                opciones_cob = ["NO (Marco N)", "Alpha Krippendorff", "Kendall W"]

        val_default = float(np.clip(azar_N, min_N, 1.0))
        target_N = st.slider("Selecciona el Nivel de Consenso (N) objetivo:", float(min_N), 1.0, val_default, 0.01, format="%.4f")
        
        if "Intervalar" in topologia: 
            perc = ni_core.calcular_percentil_universal_ni(target_N, dim_m, k_escala)
        elif "Nominal" in topologia: 
            perc = nn_core.calcular_percentil_universal_nn(target_N, dim_m, k_escala)
        else: 
            _, _, perc, _ = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala, valor_observado=target_N)

        st.info(f"💡 **El Puntero Termodinámico:** Para este diseño ($k={k_escala}, m={dim_m}$), un consenso objetivo de **$N = {target_N:.2f}$** actúa como un puntero en el **Percentil {perc:.1f}%** de la masa entrópica.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mínimo Físico", f"{min_N:.4f}")
        c2.metric("Azar Esperado", f"{azar_N:.4f}")
        c3.metric("🏆 Percentil", f"{perc:.2f} %")
        
        estimadores_cob = st.multiselect("Estimadores a enfrentar en la auditoría:", opciones_cob, default=opciones_cob)
        
        n_experimentos = st.number_input("Matrices a simular", 10, 500, 50, 10)
        carga = dim_n * dim_m * n_experimentos
        if carga > 1000000: 
            st.error(f"🚨 **Límite excedido:** {carga:,} celdas. Reduce los experimentos.")
        
        if st.button("🔬 Iniciar Prueba", type="primary", disabled=(carga > 1000000) or not estimadores_cob):
            with st.spinner("Simulando universos correlacionados..."):
                res_cob = ejecutar_auditoria_cobertura(topologia, dim_n, dim_m, k_escala, target_N, n_experimentos, replicas, estimadores_cob)
                if res_cob:
                    df_cob = pd.DataFrame(res_cob)
                    formato_columnas = {
                        "Cobertura Población (%)": "{:.1f}", "Cobertura Muestra (%)": "{:.1f}", 
                        "µ(Población Real)": "{:.4f}", "µ(Valor Muestra)": "{:.4f}", "Media Ancho IC": "{:.4f}"
                    }
                    formato_valido = {k: v for k, v in formato_columnas.items() if k in df_cob.columns}
                    st.dataframe(df_cob.style.format(formato_valido), use_container_width=True)

# ==============================================================================
# PESTAÑA 3: INFORME MULTIRRANGO
# ==============================================================================
elif st.session_state["pestaña_activa"] == '📈 Informe Multirrango':
    st.markdown("### 📈 Auditoría de Cobertura Avanzada (Informe Multirrango)")
    st.error("🛑 **AVISO CRÍTICO DE RENDIMIENTO EN LA NUBE:** Este análisis ejecuta simulaciones masivas anidadas.")
    
    st.markdown("#### 📐 Configuración de Rangos y Pasos")
    
    row1_1, row1_2, row1_3 = st.columns(3)
    with row1_1: k_min = st.number_input("Escala Mínima (k)", 2, 20, 3)
    with row1_2: k_max = st.number_input("Escala Máxima (k)", 2, 20, 5)
    with row1_3: k_step = st.number_input("Paso de Escala (Δk)", 1, 10, 1)
        
    row2_1, row2_2, row2_3 = st.columns(3)
    with row2_1: m_min = st.number_input("Jueces Mínimos (m)", 2, 50, 4)
    with row2_2: m_max = st.number_input("Jueces Máximos (m)", 2, 50, 8)
    with row2_3: m_step = st.number_input("Paso de Jueces (Δm)", 1, 10, 2)

    row3_1, row3_2, row3_3 = st.columns(3)
    with row3_1: n_min = st.number_input("Sujetos Mínimos (n)", 5, 2000, 10)
    with row3_2: n_max = st.number_input("Sujetos Máximos (n)", 5, 2000, 30)
    with row3_3: n_step = st.number_input("Paso de Sujetos (Δn)", 5, 500, 10)

    st.markdown("#### 🎲 Configuración de Cobertura y Simulación")
    
    row4_1, row4_2, row4_3 = st.columns(3)
    with row4_1: target_min = st.number_input("Consenso Mínimo Objetivo (N)", 0.0, 1.0, 0.4, 0.05)
    with row4_2: target_max = st.number_input("Consenso Máximo Objetivo (N)", 0.0, 1.0, 0.8, 0.05)
    with row4_3: target_step = st.number_input("Paso de Consenso (ΔN)", 0.01, 0.5, 0.2, 0.05)

    row5_1, row5_2 = st.columns(2)
    with row5_1: n_experimentos = st.number_input("Experimentos (Matrices por celda)", 5, 1000, 20, 5)
    with row5_2: replicas_batch = st.number_input("Réplicas del Bootstrap (S)", 50, 5000, 200, 50)

    if "Intervalar" in topologia: opciones_cob = ["NI (Marco N)", "Alpha Krippendorff", "ICC(2,1)"]
    elif "Nominal" in topologia: opciones_cob = ["NN (Marco N)", "Alpha Krippendorff", "Kappa Fleiss"]
    else: opciones_cob = ["NO (Marco N)", "Alpha Krippendorff", "Kendall W"]
    estimadores_cob = st.multiselect("Modelos a evaluar en paralelo:", opciones_cob, default=opciones_cob)

    if st.button("🔬 Iniciar Procesamiento Masivo", type="primary", disabled=not estimadores_cob):
        k_arr = range(int(k_min), int(k_max) + 1, int(k_step))
        m_arr = range(int(m_min), int(m_max) + 1, int(m_step))
        n_arr = range(int(n_min), int(n_max) + 1, int(n_step))
        t_arr = np.arange(float(target_min), float(target_max) + 0.0001, float(target_step))

        total_combinaciones = len(k_arr) * len(m_arr) * len(n_arr) * len(t_arr)
        st.info(f"Calculando {total_combinaciones:,} celdas de diseño espacial...")

        progreso = st.progress(0)
        status_text = st.empty()
        contador = 0
        lista_resultados_memoria = []

        for curr_k in k_arr:
            for curr_m in m_arr:
                for curr_n in n_arr:
                    for curr_t in t_arr:
                        contador += 1
                        status_text.text(f"Procesando celda {contador}/{total_combinaciones} ➔ (k={curr_k}, m={curr_m}, n={curr_n}, N_obj={curr_t:.4f})")
                        
                        res_celda = ejecutar_auditoria_cobertura(
                            topologia, int(curr_n), int(curr_m), int(curr_k), 
                            float(curr_t), int(n_experimentos), int(replicas_batch), estimadores_cob
                        )
                        
                        if res_celda:
                            df_celda = pd.DataFrame(res_celda)
                            
                            if "Intervalar" in topologia: p_val = ni_core.calcular_percentil_universal_ni(float(curr_t), int(curr_m), int(curr_k))
                            elif "Nominal" in topologia: p_val = nn_core.calcular_percentil_universal_nn(float(curr_t), int(curr_m), int(curr_k))
                            else: _, _, p_val, _ = no_core.analizar_termodinamica_no(int(curr_n), int(curr_m), int(curr_k), valor_observado=float(curr_t))
                            
                            df_celda.insert(0, "Percentil (%)", round(p_val, 2))
                            df_celda.insert(0, "Réplicas (S)", int(replicas_batch))
                            df_celda.insert(0, "Experimentos", int(n_experimentos))
                            df_celda.insert(0, "Nivel Cobertura Objetivo (N)", float(curr_t))
                            df_celda.insert(0, "Dimension n", int(curr_n))
                            df_celda.insert(0, "Dimension m", int(curr_m))
                            df_celda.insert(0, "Escala k", int(curr_k))
                            
                            lista_resultados_memoria.append(df_celda)

                        progreso.progress(contador / total_combinaciones)

        status_text.empty()
        
        if lista_resultados_memoria:
            df_final = pd.concat(lista_resultados_memoria, ignore_index=True)
            st.session_state['df_multirrango'] = df_final 
            csv_data = df_final.to_csv(index=False, sep=";").encode('utf-8')
            
            st.success("✨ ¡Informe multirrango completado con éxito!")
            st.download_button("📥 Descargar Informe Completo (.csv)", csv_data, "informe_cobertura_masivo.csv", "text/csv", type="primary")

    if 'df_multirrango' in st.session_state:
        df_explorador = st.session_state['df_multirrango'].copy()
        
        st.markdown("---")
        st.markdown("### 📊 Explorador Termodinámico Interactivo")
        
        col_muestra = "µ(Valor Muestra)" if "µ(Valor Muestra)" in df_explorador.columns else "µ\n(SAMPLE)"
        col_pob = "µ(Población Real)" if "µ(Población Real)" in df_explorador.columns else "µ\n(POBLATION)"
        col_target = "Nivel Cobertura Objetivo (N)" if "Nivel Cobertura Objetivo (N)" in df_explorador.columns else "TARGET \nCOVERAGE"
        col_estimador = "Estimador" if "Estimador" in df_explorador.columns else "COEFFICIENT"
        col_k = "Escala k" if "Escala k" in df_explorador.columns else "k"
        col_m = "Dimension m" if "Dimension m" in df_explorador.columns else "m"
        col_n = "Dimension n" if "Dimension n" in df_explorador.columns else "n"
        
        col_cob_pob = "Cob. Población (%)" if "Cob. Población (%)" in df_explorador.columns else ("Cobertura Población (%)" if "Cobertura Población (%)" in df_explorador.columns else "POPULATION\nCOVERAGE (%)")
        
        if col_muestra in df_explorador.columns and col_pob in df_explorador.columns:
            df_explorador["Sesgo (Bias)"] = df_explorador[col_muestra] - df_explorador[col_pob]
        
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1: metrica_sel = st.selectbox("Métrica a visualizar:", ["Sesgo (Bias)", col_cob_pob])
        with col_f2: k_sel = st.selectbox("Escala (k):", sorted(df_explorador[col_k].unique()))
        with col_f3: m_sel = st.selectbox("Jueces (m):", sorted(df_explorador[col_m].unique()))
        with col_f4: n_sel = st.selectbox("Sujetos (n):", sorted(df_explorador[col_n].unique()))

        df_filtrado = df_explorador[
            (df_explorador[col_k] == k_sel) & 
            (df_explorador[col_m] == m_sel) &
            (df_explorador[col_n] == n_sel)
        ]

        if not df_filtrado.empty:
            y_title = "Sesgo (Error Sistemático)" if metrica_sel == "Sesgo (Bias)" else "Cobertura Poblacional (%)"
            fig = px.line(
                df_filtrado, x=col_target, y=metrica_sel, color=col_estimador, markers=True,
                title=f"Evolución de {y_title} (k={k_sel}, m={m_sel}, n={n_sel})",
                color_discrete_map={
                    "NI (Marco N)": "#00CC96", "NN (Marco N)": "#00CC96", "NO (Marco N)": "#00CC96",
                    "Alpha Krippendorff": "#EF553B", "ICC(2,1)": "#636EFA", "Kappa Fleiss": "#FFA15A", "Kendall W": "#AB63FA"
                }
            )
            fig.update_layout(hovermode="x unified", xaxis=dict(title="Consenso Objetivo (Target N)"), yaxis=dict(title=y_title))
            if metrica_sel == col_cob_pob:
                fig.update_yaxes(range=[0, 105])
                fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Meta 95%")
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: INVARIANZA Y SENSIBILIDAD
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🔄 Invarianza (n) y Sensibilidad (m)':
    st.markdown("### Invarianza muestral (n) y Sensibilidad del panel (m)")
    if matriz_empirica is None or k_escala is None: 
        st.info("Sube una matriz y define k en el panel lateral.")
    else:
        tipo_invarianza = st.radio("Selecciona la dimensión de escalado:", ["↕️ Vertical (Multiplicar Sujetos - n)", "↔️ Horizontal (Multiplicar Jueces - m)"], horizontal=True)
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        
        with c1:
            if "Vertical" in tipo_invarianza:
                factor = st.number_input("Multiplicador (X veces):", 2, 100000, 2)
                bloquear_test = False
            else:
                max_factor = int(50 // m_jueces)
                if max_factor < 2:
                    st.warning(f"⚠️ Tu matriz inicial ya tiene {m_jueces} jueces.")
                    factor = 1
                    bloquear_test = True
                else:
                    factor = st.number_input("Multiplicador (X veces):", 2, max_factor, 2)
                    bloquear_test = False

        with c2: replicas_inv = st.number_input("Réplicas (Velocidad):", 10, 500, 50)
            
        if st.button("🔄 Ejecutar Test Bidimensional", type="primary", disabled=bloquear_test):
            m_rep = np.tile(matriz_empirica, (factor, 1)) if "Vertical" in tipo_invarianza else np.tile(matriz_empirica, (1, factor))
            c_orig = Counter(tuple(x) for x in matriz_empirica)
            c_rep = Counter(tuple(x) for x in m_rep)
            res_inv = []
            
            with st.spinner("Calculando huella topológica..."):
                def extr(func, *args):
                    res = func(*args)
                    if isinstance(res, dict): 
                        val = res.get('Muestra', res.get('AKI Muestra', res.get('ICC Muestra', res.get('AKN Muestra', res.get('KF Muestra', res.get('AKO Muestra', res.get('W Muestra', 0)))))))
                        inf, sup = res.get('IC Inf', 0), res.get('IC Sup', 0)
                    else: val, inf, sup = res[0], res[2], res[3]
                    return val, inf, sup, (sup - inf)

                def add_res(nom, func, arg_o, arg_r, *extra):
                    try:
                        v_o, i_o, s_o, a_o = extr(func, arg_o, *extra)
                        v_r, i_r, s_r, a_r = extr(func, arg_r, *extra)
                        res_inv.append({"Estimador": nom, "Orig": v_o, "IC O": f"[{i_o:.3f}, {s_o:.3f}]", "Ancho O": a_o, "Repl": v_r, "IC R": f"[{i_r:.3f}, {s_r:.3f}]", "Ancho R": a_r})
                    except Exception as e: st.error(f"Error procesando {nom}: {e}")

                if "Intervalar" in topologia:
                    add_res("N-Interval (NI)", ni_core.calcular_estadisticas_ni_unificada, c_orig, c_rep, k_escala, replicas_inv)
                    add_res("Alpha Krippendorff", aki_core.calcular_estadisticas_aki, matriz_empirica, m_rep, replicas_inv)
                    add_res("ICC(2,1)", icc21_core.calcular_estadisticas_icc21, matriz_empirica, m_rep)
                elif "Nominal" in topologia:
                    add_res("N-Nominal (NN)", nn_core.calcular_estadisticas_nn_unificada, c_orig, c_rep, k_escala, replicas_inv)
                    add_res("AKN", akn_core.calcular_estadisticas_akn, matriz_empirica, m_rep, replicas_inv)
                    add_res("Kappa Fleiss", kf_core.calcular_estadisticas_kf, matriz_empirica, m_rep, replicas_inv)
                else:
                    add_res("N-Ordinal (NO)", no_core.calcular_estadisticas_no_unificada, c_orig, c_rep, k_escala, replicas_inv)
                    add_res("AKO", ako_core.calcular_estadisticas_ako, matriz_empirica, m_rep, replicas_inv)
                    add_res("Kendall W", w_core.calcular_estadisticas_w, matriz_empirica, m_rep, replicas_inv)

            if res_inv:
                df_inv = pd.DataFrame(res_inv)
                df_inv["Var"] = df_inv["Repl"] - df_inv["Orig"]
                st.dataframe(df_inv.style.format({"Orig": "{:.4f}", "Repl": "{:.4f}", "Ancho O": "{:.4f}", "Ancho R": "{:.4f}", "Var": "{:+.4f}"}), use_container_width=True)

# ==============================================================================
# PESTAÑA 5: GENERADOR DE MATRICES SINTÉTICAS
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🏗️ Generador de Matrices':
    st.subheader("🏗️ Generador de Matrices Sintéticas")
    st.error("###### ⚠️ Si tiene cargada una matriz, bórrela antes de proceder.")
    
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1: gen_k = st.number_input("Escala (k)", 2, 20, 5, key="gk")
    with g_col2: gen_m = st.number_input("Jueces (m)", 2, 50, 7, key="gm")
    with g_col3: gen_n = st.number_input("Sujetos (n)", 1, 5000000, 15, key="gn")

    g_col4, g_col5 = st.columns(2)
    with g_col4: gen_faltantes = st.slider("Faltantes (%)", 0.0, 100.0, 0.0, 5.0) / 100
    with g_col5: gen_decimales = st.slider("Decimales (Ruido)", 0, 2, 0)

    st.markdown("#### Patrón Base (Norma)")
    if "df_base_1" not in st.session_state or st.session_state.get("gk_last_base1") != gen_k:
        st.session_state["df_base_1"] = pd.DataFrame(np.full((1, gen_k), round(100/gen_k, 2)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=["Prob (%)"])
        st.session_state["gk_last_base1"] = gen_k
    edit_base_1 = st.data_editor(st.session_state["df_base_1"], use_container_width=True, key="edit_base_1")

    modo = st.radio("Selecciona el modo de patrón:", ["Patrón global", "Patrón por sujeto"], horizontal=True, label_visibility="collapsed")
    
    edit_n_rows = None
    if modo == "Patrón por sujeto":
        base_hash = hash(edit_base_1.values.tobytes())
        if st.session_state.get("last_state_key") != (gen_n, gen_k, base_hash):
            st.session_state["df_n_rows"] = pd.DataFrame(np.tile(edit_base_1.values, (gen_n, 1)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=[f"S{i+1}" for i in range(gen_n)])
            st.session_state["last_state_key"] = (gen_n, gen_k, base_hash)
        edit_n_rows = st.data_editor(st.session_state["df_n_rows"], use_container_width=True)

    st.markdown("#### Excepciones (Agujeros Negros)")
    excepciones = st.multiselect("Selecciona sujetos para inyectar anomalías:", [f"S{i+1}" for i in range(gen_n)])
    edit_exc = None
    if excepciones:
        if st.session_state.get("exc_last") != excepciones or st.session_state.get("exc_last_k") != gen_k:
            st.session_state["df_exc"] = pd.DataFrame(np.full((len(excepciones), gen_k), round(100/gen_k, 2)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=excepciones)
            st.session_state["exc_last"] = excepciones
            st.session_state["exc_last_k"] = gen_k
        edit_exc = st.data_editor(st.session_state["df_exc"], use_container_width=True)

    def crear_matriz_sintetica():
        base_p = edit_base_1.iloc[0].values
        p_norm = base_p / base_p.sum() if base_p.sum() > 0 else np.ones(gen_k) / gen_k
        
        if modo == "Patrón global" and not excepciones:
            matriz = np.random.choice(np.arange(1, gen_k + 1, dtype=float), size=(gen_n, gen_m), p=p_norm)
            if gen_decimales > 0:
                matriz = np.clip(matriz + np.random.uniform(-0.1, 0.1, size=(gen_n, gen_m)), 1.0, float(gen_k))
                matriz = np.round(matriz, gen_decimales)
            if gen_faltantes > 0:
                matriz[np.random.rand(gen_n, gen_m) < gen_faltantes] = np.nan
            return pd.DataFrame(matriz, columns=[f"J{i+1}" for i in range(gen_m)])

        matriz = np.empty((gen_n, gen_m), dtype=float)
        exc_dict = {s_id: edit_exc.loc[s_id].values for s_id in excepciones} if (edit_exc is not None and excepciones) else {}
        n_rows_values = edit_n_rows.values if (modo == "Patrón por sujeto" and edit_n_rows is not None) else None

        for i in range(gen_n):
            sujeto_id = f"S{i+1}"
            p = exc_dict[sujeto_id] if sujeto_id in exc_dict else (n_rows_values[i] if n_rows_values is not None else base_p)
            p_n = p / p.sum() if p.sum() > 0 else np.ones(gen_k)/gen_k
            matriz[i] = np.random.choice(np.arange(1, gen_k + 1, dtype=float), size=gen_m, p=p_n)

        if gen_decimales > 0:
            matriz = np.clip(matriz + np.random.uniform(-0.1, 0.1, size=(gen_n, gen_m)), 1.0, float(gen_k))
            matriz = np.round(matriz, gen_decimales)
        if gen_faltantes > 0:
            matriz[np.random.rand(gen_n, gen_m) < gen_faltantes] = np.nan

        return pd.DataFrame(matriz, columns=[f"J{i+1}" for i in range(gen_m)])

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Preparar Descarga", use_container_width=True):
            st.session_state['matriz_csv_temp'] = crear_matriz_sintetica()
        if 'matriz_csv_temp' in st.session_state:
            st.download_button("Descargar CSV", st.session_state['matriz_csv_temp'].to_csv(index=False), "matriz_sintetica.csv", "text/csv", use_container_width=True)

    with col_b2:
        if st.button("🚀 Generar y Cargar en la App", type="primary", use_container_width=True):
            with st.spinner("Sintetizando..."):
                df_sintetico = crear_matriz_sintetica()
                import io
                buffer = io.StringIO()
                df_sintetico.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                conteo_est, n_tot = cargar_y_agregar_dataset(buffer, sep=",")
                st.session_state.pop('res_inferencia', None)
                st.session_state.update({
                    'usar_sintetica': True, 'matriz_generada_app': df_sintetico, 'df_original_raw': df_sintetico, 
                    'df': df_sintetico, 'diccionario_estados': conteo_est, 'n_total': n_tot, 'k_escala': gen_k,
                    'topologia_activa': 'Intervalar (Discreta/Continua)', 'pestaña_activa': '📊 Cálculo del Consenso'
                })
                st.rerun()

# ==============================================================================
# PESTAÑA 6: DUELO DE PRECISIÓN N vs CLÁSICOS
# ==============================================================================
elif st.session_state["pestaña_activa"] == '⚔️ Duelo: N vs Clásicos':
    st.markdown("### ⚔️ Duelo de Precisión: N vs Clásicos")
    if k_escala is None:
        st.warning("⚠️ Debes definir el límite del hiperespacio (k) en el panel lateral.")
    else:
        escenarios = {
            "Libre (Configuración Manual)": {},
            "1. Paradoja de la Varianza Cero (Techo)": {"n": 50, "m": 5, "target": 0.96, "desc": "Fuerza un consenso casi absoluto."},
            "2. Muestras Pequeñas (Clínico/Expertos)": {"n": 8, "m": 3, "target": 0.65, "desc": "Simula un panel de expertos."},
            "3. Ruido y Dispersión (Caos)": {"n": 100, "m": 5, "target": 0.15, "desc": "Inyecta alta entropía."},
            "4. Gran Escala (Stress Test Big Data)": {"n": 1500, "m": 10, "target": 0.50, "desc": "Simula un entorno de alto volumen."}
        }
        
        escenario_sel = st.selectbox("Selecciona el tipo de auditoría:", list(escenarios.keys()))
        if escenario_sel != "Libre (Configuración Manual)":
            st.info(f"💡 **Objetivo:** {escenarios[escenario_sel]['desc']}")
            dim_n, dim_m, target_preset, modo_libre = escenarios[escenario_sel]["n"], escenarios[escenario_sel]["m"], escenarios[escenario_sel]["target"], False
        else:
            dim_n, dim_m, target_preset, modo_libre = (n_sujetos if matriz_empirica is not None else 50), (m_jueces if matriz_empirica is not None else 7), None, True

        with st.spinner("Calculando fronteras..."):
            if "Intervalar" in topologia: azar_N, min_N = ni_core.calcular_azar_termodinamico_ni(dim_m, k_escala), 0.0
            elif "Nominal" in topologia: azar_N, min_N = nn_core.calcular_azar_termodinamico_nn(dim_m, k_escala), np.sqrt(min(nn_core._build_macrostate_dictionary_nn(dim_m, k_escala).keys()))
            else: azar_N, _, _, min_N = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1: n_duelo = st.slider("Experimentos", 10, 200, 50)
        with col_d2:
            val_default = float(np.clip(target_preset if not modo_libre else azar_N, min_N, 1.0))
            target_N = st.slider("Consenso Objetivo:", float(min_N), 1.0, val_default, 0.01, format="%.4f", disabled=not modo_libre)
        
        if st.button("🔥 Iniciar Combate de Precisión", type="primary"):
            with st.spinner("Simulando universos..."):
                resultados = ejecutar_duelo_ia(topologia, dim_n, dim_m, k_escala, target_N, n_duelo, replicas)
                if resultados:
                    df_duel = pd.DataFrame(resultados)
                    cols_cobertura = [c for c in df_duel.columns if "cobertura" in c.lower() or "cob" in c.lower()]
                    df_duel = df_duel.drop(columns=cols_cobertura, errors='ignore')
                    
                    c1, c2 = st.columns(2)
                    try:
                        col_mse = [c for c in df_duel.columns if "mse" in c.lower()][0]
                        col_bias = [c for c in df_duel.columns if "bias" in c.lower() or "sesgo" in c.lower()][0]
                        n_mse, a_mse = float(df_duel.iloc[0][col_mse]), float(df_duel.iloc[1][col_mse])
                        n_bias, a_bias = float(df_duel.iloc[0][col_bias]), float(df_duel.iloc[1][col_bias])
                        
                        c1.metric("Precisión N (MSE)", f"{n_mse:.6f}", f"{n_mse - a_mse:+.6f} vs Clásico", delta_color="inverse")
                        c2.metric("Sesgo N (Bias)", f"{n_bias:.6f}", f"{abs(n_bias) - abs(a_bias):+.6f} magnitud vs Clásico", delta_color="inverse")
                    except Exception: pass
                    
                    st.dataframe(df_duel, use_container_width=True)

# ==============================================================================
# PESTAÑA 7: BASE DE DATOS DE DICCIONARIOS Y FIRMAS
# ==============================================================================
elif st.session_state.get("pestaña_activa") == '📚 BD Diccionarios':
    st.markdown("### 🧬 Análisis de Clases de Equivalencia y Firmas")
    st.write("Caracterización del espacio de configuraciones $(k, m)$ a partir de las firmas observadas.")
    
    dict_activo = st.session_state.get("diccionario_estados")
    
    if dict_activo:
        primer_vector = next(iter(dict_activo.keys()))
        m = len(primer_vector)
        
        # 1. Intento de carga directa desde JSON precalculado en carpeta data/
        firmas_json = cargar_firmas_precalculadas(m)
        
        if firmas_json is not None:
            st.success(f"⚡ Carga ultrarrápida: Leyendo firmas teóricas precalculadas para $m={m}$ desde disco.")
            k = firmas_json.get("k", len(set().union(*dict_activo.keys())))
            df_firmas = pd.DataFrame(firmas_json.get("firmas", []))
            u_micro = len(dict_activo)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dimensión m (Evaluadores)", m)
            col2.metric("Categorías k Observadas", k)
            col3.metric("Clases / Firmas Activas", len(df_firmas))
            col4.metric("Microestados Únicos", u_micro)
            
            st.markdown("---")
            st.markdown("#### 📊 Distribución de Clases de Equivalencia (Firmas)")
            st.dataframe(df_firmas, use_container_width=True, hide_index=True)
            
        else:
            # 2. Fallback procesado ligero si no existe JSON precalculado
            st.info(f"Procesando firmas en tiempo de ejecución para $m={m}$...")
            todas_categorias = set()
            for vec in dict_activo.keys():
                todas_categorias.update(vec)
            k = len(todas_categorias)
            
            firmas_dict = {}
            for vec, freq in dict_activo.items():
                firma = calcular_firma(vec)
                vec_limpio = str(tuple(int(x) for x in vec)) if isinstance(vec, (tuple, list, np.ndarray)) else str(vec)
                    
                if firma not in firmas_dict:
                    firmas_dict[firma] = {
                        "microestados_unicos": 0,
                        "frecuencia_total": 0,
                        "ejemplo_microestado": vec_limpio
                    }
                firmas_dict[firma]["microestados_unicos"] += 1
                firmas_dict[firma]["frecuencia_total"] += freq

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dimensión m (Evaluadores)", m)
            col2.metric("Categorías k Observadas", k)
            col3.metric("Clases / Firmas Activas", len(firmas_dict))
            col4.metric("Microestados Únicos", len(dict_activo))
            
            st.markdown("---")
            st.markdown("#### 📊 Distribución de Clases de Equivalencia (Firmas)")
            
            filas_firmas = []
            N_total = sum(d["frecuencia_total"] for d in firmas_dict.values())
            
            for firma, datos in sorted(firmas_dict.items(), key=lambda x: x[1]["frecuencia_total"], reverse=True):
                filas_firmas.append({
                    "Firma / Clase (Partición de m)": str(firma),
                    "Multiplicidad Observada (Microestados)": datos["microestados_unicos"],
                    "Frecuencia Acumulada (n)": datos["frecuencia_total"],
                    "Proporción (p)": round(datos['frecuencia_total'] / N_total, 4) if N_total > 0 else 0,
                    "Ejemplo de Vector": datos["ejemplo_microestado"]
                })
                
            df_firmas = pd.DataFrame(filas_firmas)
            st.dataframe(df_firmas, use_container_width=True, hide_index=True)
            
            with st.expander("🔍 Ver todos los microestados agrupados por Firma"):
                for firma, datos in firmas_dict.items():
                    st.markdown(f"**Firma {firma}:** {datos['microestados_unicos']} microestados | Frecuencia total: {datos['frecuencia_total']}")
                
    else:
        st.info("No hay ningún diccionario de estados cargado en la sesión actual. Carga un dataset desde la barra lateral.")

# ==============================================================================
# PESTAÑA 8: MANUAL DE USUARIO
# ==============================================================================
elif st.session_state["pestaña_activa"] == '📖 Manual de Usuario':
    st.write("") 
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.video("https://youtu.be/mHQekxmCxH4")
        
    with col3:
        st.write(""); st.write(""); st.write(""); st.write("")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        pdf_path = os.path.join(base_dir, "Doc", "N-Metrics_Logic_Blueprint.pdf")
        
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar N-Metrics Logic Blueprint (PDF)",
                    data=pdf_file,
                    file_name="N-Metrics_Logic_Blueprint.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.error("⚠️ El manual PDF no se encuentra disponible en el servidor.")
            st.info(f"🔍 [Depuración] El sistema está buscando el archivo exactamente aquí: `{pdf_path}`")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.video("https://youtu.be/BjfPvSKJeXA")

    st.markdown("## 📖 Manual de Usuario y Fundamentos Teóricos")
    st.write("Bienvenido al entorno analítico de **Métricas N**. Esta plataforma permite evaluar el nivel de consenso real de matrices empíricas, superando las paradojas de los estimadores frecuentistas clásicos mediante la aplicación de la **Termodinámica Exacta de la Información**.")
    
    st.markdown("---")
    st.markdown("### 1. El Motor Termodinámico: El Valor Poblacional")
    st.write("La mayor innovación de esta plataforma radica en cómo descubre la **Verdad Asintótica** (Pob. Real) de tus datos.")
    st.markdown("""
    * **1. Extracción de Macroestados:** El algoritmo aísla los patrones topológicos puros de tu matriz (las firmas de consenso).
    * **2. Multiplicidad Teórica ($\\Omega$):** Utilizando combinatoria exacta, calcula el 'volumen' geométrico que ocupa cada patrón.
    * **3. Proyección Asintótica:** Pondera las frecuencias de tu muestra empírica cruzándolas con su volumen teórico.
    """)

    st.markdown("---")
    st.markdown("### 2. Carga de Datos y Topología (Panel Lateral)")
    st.write("El primer paso es suministrar la materia prima y definir las leyes físicas del hiperespacio de probabilidad.")

    st.markdown("---")
    st.markdown("### 3. Generador de Matrices Sintéticas (Pestaña 4)")
    st.write("Crea matrices termodinámicas a medida para probar las capacidades del sistema o generar casos de estudio controlados.")

    st.markdown("---")
    st.markdown("### 4. Inferencia de Consenso: Marco N vs Clásicos (Pestaña 1)")
    st.write("Esta sección somete tu matriz a una batalla de estimadores.")

    st.markdown("---")
    st.markdown("### 5. Auditoría Estructural: Escáner de Anomalías")
    st.write("Identifica qué elementos de la matriz están inyectando entropía excesiva en el sistema:")
    st.latex(r"Límite = \mu_{local} - (\text{Umbral} \cdot \sigma_{local})")

    st.markdown("---")
    st.markdown("### 6. Pruebas de Estrés y Diagnóstico (Pestañas 2 y 3)")
    st.write("Herramientas avanzadas para auditar la honestidad matemática de los estimadores.")

    st.markdown("---")
    st.markdown("### ⚔️ 7. Guía del Duelo de Titanes (N vs. Clásicos - Pestaña 6)")
    st.write("Esta pestaña es un laboratorio de estrés estadístico para auditar la precisión en condiciones críticas de Inteligencia Artificial.")

    st.markdown("---")
    st.markdown("### 🧬 8. Análisis de Clases de Equivalencia y Base de Datos (Pestaña 7)")
    st.write("Escáner del espacio de fases de la aplicación, abstrayendo la matriz desde sus microestados hacia sus firmas combinatorias.")

# ==============================================================================
# PESTAÑA 9: AUTORÍA Y LICENCIA
# ==============================================================================
elif st.session_state["pestaña_activa"] == '📜 Autoría y Licencia':
    st.markdown("### Código Abierto y Transparencia")
    st.write("El Marco Termodinámico $N$ es un proyecto de ciencia abierta creado por Manuel Narbona Sarria.")
    
    st.markdown("---")
    col_git, col_pypi = st.columns(2)
    with col_git:
        st.markdown("#### 🐙 Repositorio en GitHub")
        st.write("Audita los teoremas matemáticos y el motor de simulación.")
        st.link_button("Ver en GitHub", "https://github.com/manuel-narsa/n-metrics")
    with col_pypi:
        st.markdown("#### 🐍 Paquete Oficial en PyPI")
        st.write("Integra N-Metrics en tus propios scripts de Python.")
        st.code("pip install n-metrics", language="bash")
        st.link_button("Ver en PyPI", "https://pypi.org/project/n-metrics/")
        
    st.markdown("---")
    st.markdown("### Licencia de Uso")
    st.write("**Licencia Abierta**. GNU GPLv3.")
    
    st.markdown("---")
    st.markdown("### 🎓 Cómo citar este trabajo")
    st.write("Si utilizas N-Metrics o el Marco Termodinámico N en tu investigación, por favor cita el manuscrito original:")
    
    st.info("Narbona-Sarria, M. (2026). N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus. Available at Zenodo: https://doi.org/10.5281/zenodo.20075068")
    st.info("Narbona-Sarria, M. (2026). N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus. Available at SSRN: https://ssrn.com/abstract=7119419 or http://dx.doi.org/10.2139/ssrn.7119419")
    
    with st.expander("Ver formato BibTeX para LaTeX"):
        st.code("""@article{narbona_n_coefficient,
  title={N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus},
  author={Narbona Sarria, Manuel},
  journal={Preprint},
  year={2026},
  url={https://doi.org/10.5281/zenodo.20075068}
  url={http://dx.doi.org/10.2139/ssrn.7119419}
}""", language="bibtex")