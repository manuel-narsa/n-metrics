import streamlit as st
import pandas as pd
import numpy as np
import time
import itertools
import math
import os
import sys  # <-- AÑADE ESTO
from PIL import Image
from collections import defaultdict, Counter

# ==============================================================================
# FIJAR RUTA PRINCIPAL (Soluciona el ModuleNotFoundError)
# ==============================================================================
# Esto le dice a Python: "La carpeta donde está app.py es la raíz del proyecto"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# IMPORTACIONES DEL MOTOR TERMODINÁMICO (N-Metrics)
# ==============================================================================
# 1. Los núcleos matemáticos
from nmetrics.interval import ni_core, aki_core, icc21_core
from nmetrics.nominal import nn_core, akn_core, kf_core
from nmetrics.ordinal import no_core, ako_core, w_core

# 2. Las herramientas modulares (Ahora importadas DESDE nmetrics)
from nmetrics.utils.data_handler import reset_session_state, cargar_y_agregar_dataset, procesar_datos_df, validar_condiciones_analisis
from nmetrics.utils.simulations import generar_matriz_termodinamica_exacta, ejecutar_auditoria_cobertura, ejecutar_duelo_ia
from nmetrics.utils.anomalies import escaner_anomalias_ultra_rapido

# ... (sigue el resto de tu código) ...
# ==============================================================================
# MOTOR DE INFERENCIA DE UI (Debe quedarse en app.py por su dependencia con Streamlit)
# ==============================================================================
def ejecutar_calculo_optimizado(dict_estados, topologia, k_calc, replicas, estimadores, m_jueces, n_total, matriz_np):
    """Enrutador principal de cálculos que alimenta las tablas de la interfaz."""
    resultados_inf = []
    
    # --- A. MOTOR UNIFICADO: NI (Intervalar) ---
    if "Intervalar" in topologia and "NI (Marco N)" in estimadores:
        try:
            ni_muestra, pob_real, inf, sup = ni_core.calcular_estadisticas_ni_unificada(dict_estados, k_calc, replicas)
            p_e = ni_core.calcular_azar_termodinamico_ni(m_jueces, k_calc)
            perc = ni_core.calcular_percentil_universal_ni(pob_real, m_jueces, k_calc)
            resultados_inf.append({
                "Métrica": "NI", "Muestra": ni_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": p_e, "Percentil (%)": perc, "Motor": "Termodinámica Condensada (SP)"
            })
        except Exception as e: st.warning(f"⚠️ El motor NI falló: {e}")

    # --- B. MOTOR UNIFICADO: NN (Nominal) ---
    elif "Nominal" in topologia and "NN (Marco N)" in estimadores:
        try:
            nn_muestra, pob_real, inf, sup = nn_core.calcular_estadisticas_nn_unificada(dict_estados, k_calc, replicas)
            p_e = nn_core.calcular_azar_termodinamico_nn(m_jueces, k_calc)
            perc = nn_core.calcular_percentil_universal_nn(pob_real, m_jueces, k_calc)
            resultados_inf.append({
                "Métrica": "NN", "Muestra": nn_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": p_e, "Percentil (%)": perc, "Motor": "Termodinámica Condensada (SP)"
            })
        except Exception as e: st.warning(f"⚠️ El motor NN falló: {e}")

    # --- C. MOTOR UNIFICADO: NO (Ordinal) ---
    elif "Ordinal" in topologia and "NO (Marco N)" in estimadores:
        try:
            no_muestra, pob_real, inf, sup = no_core.calcular_estadisticas_no_unificada(dict_estados, k_calc, replicas)
            azar_no, _, perc_no, _ = no_core.analizar_termodinamica_no(n_total, m_jueces, k_calc, valor_observado=pob_real)
            resultados_inf.append({
                "Métrica": "NO", "Muestra": no_muestra, "Pob. Real": pob_real, 
                "IC Inf": inf, "IC Sup": sup, "Ancho IC": sup - inf,
                "Valor Azar": azar_no, "Percentil (%)": perc_no, "Motor": "Termodinámica Geométrica"
            })
        except Exception as e: st.warning(f"⚠️ El motor NO falló: {e}")

    # --- D. ESTIMADORES CLÁSICOS (Fuerza Bruta) ---
    if n_total <= 50000:
        if "Intervalar" in topologia:
            if "AKI (Bootstrap C.)" in estimadores:
                try:
                    p_aki = aki_core.calcular_aki_poblacion_asintotica(matriz_np, 1, k_calc, 100)
                    st_aki = aki_core.calcular_estadisticas_aki(matriz_np, replicas)
                    resultados_inf.append({"Métrica": "AKI", "Muestra": st_aki['AKI Muestra'], "Pob. Real": p_aki, "IC Inf": st_aki['IC Inf'], "IC Sup": st_aki['IC Sup'], "Motor": "Bootstrap (Resampling)"})
                except: pass
            if "ICC(2,1) (F-ANOVA)" in estimadores:
                try:
                    p_icc = icc21_core.calcular_icc_poblacion_asintotica(matriz_np, 1, k_calc, 100)
                    st_icc = icc21_core.calcular_estadisticas_icc21(matriz_np)
                    resultados_inf.append({"Métrica": "ICC(2,1)", "Muestra": st_icc['ICC Muestra'], "Pob. Real": p_icc, "IC Inf": st_icc['IC Inf'], "IC Sup": st_icc['IC Sup'], "Ancho IC": st_icc['IC Sup']-st_icc['IC Inf'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "F-ANOVA (Paramétrico)"})
                except: pass    

        if "Nominal" in topologia:
            if "AKN (Bootstrap C.)" in estimadores:
                try:
                    p_akn = akn_core.calcular_akn_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_akn = akn_core.calcular_estadisticas_akn(matriz_np, replicas)
                    resultados_inf.append({"Métrica": "AKN", "Muestra": st_akn['AKN Muestra'], "Pob. Real": p_akn, "IC Inf": st_akn['IC Inf'], "IC Sup": st_akn['IC Sup'], "Motor": "Bootstrap (Resampling)"})
                except: pass
            if "Kappa Fleiss (Bootstrap C.)" in estimadores:
                try:
                    p_kf = kf_core.calcular_kf_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_kf = kf_core.calcular_estadisticas_kf(matriz_np, replicas)
                    resultados_inf.append({"Métrica": "KF", "Muestra": st_kf['KF Muestra'], "Pob. Real": p_kf, "IC Inf": st_kf['IC Inf'], "IC Sup": st_kf['IC Sup'], "Motor": "Bootstrap (Resampling)"})
                except: pass

        if "Ordinal" in topologia:
            if "AKO (Bootstrap C.)" in estimadores:
                try:
                    p_ako = ako_core.calcular_ako_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_ako = ako_core.calcular_estadisticas_ako(matriz_np, replicas)
                    resultados_inf.append({"Métrica": "AKO", "Muestra": st_ako['AKO Muestra'], "Pob. Real": p_ako, "IC Inf": st_ako['IC Inf'], "IC Sup": st_ako['IC Sup'], "Motor": "Bootstrap (Resampling)"})
                except: pass
            if "Kendall W (Bootstrap C.)" in estimadores:
                try:
                    p_w = w_core.calcular_w_poblacion_asintotica(matriz_np, k_calc, 100)
                    st_w = w_core.calcular_estadisticas_w(matriz_np, replicas)
                    resultados_inf.append({"Métrica": "W", "Muestra": st_w['W Muestra'], "Pob. Real": p_w, "IC Inf": st_w['IC Inf'], "IC Sup": st_w['IC Sup'], "Motor": "Bootstrap (Resampling)"})
                except: pass

    return resultados_inf

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA (Streamlit)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_icono = os.path.join(BASE_DIR, "icono.png")

try:
    if os.path.exists(ruta_icono):
        img_icono = Image.open(ruta_icono)
        st.set_page_config(page_title="Métricas N: Termodinámica del Consenso", page_icon=img_icono, layout="wide")
    else:
        st.set_page_config(page_title="Métricas N: Termodinámica del Consenso", layout="wide")
except:
    pass

st.markdown("""
    <style>
        .block-container { padding-bottom: 10rem !important; }
    </style>
""", unsafe_allow_html=True)

# Inicialización de variables globales
if "topologia_activa" not in st.session_state: st.session_state["topologia_activa"] = "Intervalar (Continua)"
if "pestaña_activa" not in st.session_state: st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
topologia = st.session_state["topologia_activa"].split(" ")[0]

# ==============================================================================
# PANEL LATERAL (SIDEBAR)
# ==============================================================================
st.sidebar.header("📂 Carga de Datos")
separador = st.sidebar.selectbox("Separador del CSV", [",", ";", "\t"], key="sep_main", on_change=reset_session_state)
archivo_subido = st.sidebar.file_uploader("Sube tu matriz empírica (CSV)", type=["csv", "txt"], on_change=reset_session_state)

if archivo_subido is not None and 'diccionario_estados' not in st.session_state:
    with st.sidebar.status("⚙️ Procesando matriz...", expanded=True) as status:
        try:
            dict_est, n_total = cargar_y_agregar_dataset(archivo_subido, separador)
            st.session_state.diccionario_estados = dict_est
            st.session_state.n_total = n_total
            st.session_state["topologia_activa"] = "Intervalar (Continua)"
            st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
            
            archivo_subido.seek(0)
            df_raw_loaded = pd.read_csv(archivo_subido, sep=separador, header=None)
            df_raw_loaded = df_raw_loaded.replace({',': '.'}, regex=True).apply(pd.to_numeric, errors='coerce')
            st.session_state.df_original_raw = df_raw_loaded
            
            status.update(label="✅ Matriz procesada.", state="complete")
            st.rerun() 
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")

if 'df_original_raw' in st.session_state and st.session_state['df_original_raw'] is not None and len(st.session_state['df_original_raw']) > 0:
    st.sidebar.success(f"✅ Matriz activa: {st.session_state.get('n_total', 0):,} sujetos.")
    if st.session_state.get('usar_sintetica'):
        if st.sidebar.button("🗑️ Borrar Matriz Generada", use_container_width=True):
            reset_session_state()
            st.session_state["topologia_activa"] = "Intervalar (Continua)"
            st.session_state["pestaña_activa"] = "📊 Cálculo del Consenso"
            st.rerun()

st.sidebar.header("🗺️ Configuración de Escala")
v_min_user = st.sidebar.number_input("Valor MÍNIMO", value=1.0)
v_max_user = st.sidebar.number_input("Valor MÁXIMO", value=5.0)
replicas = st.sidebar.slider("Réplicas de Simulación (S)", 100, 5000, 1000)

topologia_opciones = ["Intervalar (Continua)", "Nominal (Categórica)", "Ordinal (Ordenada)"]
indice_actual = topologia_opciones.index(st.session_state["topologia_activa"]) if st.session_state["topologia_activa"] in topologia_opciones else 0
seleccion_user = st.sidebar.radio("Naturaleza de los datos:", topologia_opciones, index=indice_actual)

if seleccion_user != st.session_state["topologia_activa"]:
    st.session_state["topologia_activa"] = seleccion_user
    st.session_state.pop('res_inferencia', None)
    st.rerun()

# ==============================================================================
# PROCESAMIENTO TOPOLÓGICO Y VARIABLES GLOBALES
# ==============================================================================
matriz_original, matriz_empirica = None, None  
k_escala = st.session_state.get('k_escala', 5)            
n_sujetos, m_jueces = 0, 0

if st.session_state.get('usar_sintetica') and st.session_state.get('matriz_generada_app') is not None:
    df_gen = st.session_state['matriz_generada_app']
    n_sujetos, m_jueces = df_gen.shape
    matriz_original = df_gen.values
    matriz_empirica = df_gen.values 
elif 'df_original_raw' in st.session_state and not st.session_state.get('usar_sintetica'):
    try:
        matriz_original = st.session_state['df_original_raw'].values
        n_sujetos, m_jueces = matriz_original.shape
        magnitude = 10 ** math.floor(math.log10(abs(v_min_user))) if v_min_user != 0 else 1.0
        shift = (v_min_user / magnitude) - 1
        matriz_empirica = (matriz_original / magnitude) - shift
        
        st.session_state['matriz_empirica'] = matriz_empirica
        k_escala = int(round((v_max_user / magnitude) - shift))
        st.session_state['k_escala'] = k_escala
        
        if np.nanmax(matriz_empirica) > k_escala + 0.01 or np.nanmin(matriz_empirica) < 0.99:
            st.sidebar.error(f"🚨 Escala Inconsistente: La matriz contiene respuestas fuera del rango.")
            st.stop()
    except Exception as e: st.sidebar.error(f"Error en el procesamiento: {e}")

# ==============================================================================
# CABECERA Y NAVEGACIÓN
# ==============================================================================
col_logo, col_titulo = st.columns([1, 15], vertical_alignment="center") 
with col_logo:
    try: st.image("icono.png", width=60)
    except: pass
with col_titulo:
    st.markdown('<h1 style="margin-top: 0rem; padding-top: 0rem;">Métricas N: La Termodinámica del Consenso</h1>', unsafe_allow_html=True)
st.markdown("Plataforma oficial para la inferencia termodinámica y auditoría topológica de matrices empíricas.")

opciones_pestañas = ["📊 Cálculo del Consenso", "🎯 Pruebas de Cobertura", "🔄 Invarianza Topológica", "🏗️ Generador de Matrices", "⚔️ Duelo: N vs todos los demás", "📖 Manual de Usuario", "📜 Autoría y Licencia"]
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
        st.success(f"🚀 Análisis Activo: Procesando {n_sujetos:,} registros.")
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
                except: pass
                return estilos

            st.dataframe(df_res[columnas_a_mostrar].style.format(formatos).apply(auditar, axis=1), use_container_width=True)
            
        st.markdown("---")
        st.markdown(f"### 🔍 Escáner de Anomalías ({'Jueces Disidentes' if 'Ordinal' in topologia else 'Macroestados'})")
        umbral_sigma = st.slider("Umbral de tolerancia ($\sigma$):", 0.5, 3.0, 1.5, 0.1)

        if st.button("🔎 Ejecutar Escáner de Anomalías", type="secondary"):
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
                        df_visual = df_anomalos[['Acuerdo_Local']].rename(columns={'Acuerdo_Local': 'Acuerdo Lineal (A)'})
                        st.dataframe(df_visual.style.format("{:.4f}").apply(lambda x: ['background-color: #fee2e2; color: #dc3545; font-weight: bold;']*len(x), axis=1), use_container_width=True)
                    else: st.success(f"✨ Sistema estable bajo $\sigma = {umbral_sigma}$.")

# ==============================================================================
# PESTAÑA 2: PRUEBAS DE COBERTURA
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🎯 Pruebas de Cobertura':
    st.markdown("### Auditoría de Paradoja de Cobertura (Stress Test Global)")
    if k_escala is None: st.warning("⚠️ Debes definir el límite (k) en el panel lateral.")
    else:
        dim_n, dim_m = n_sujetos if matriz_empirica is not None else 50, m_jueces if matriz_empirica is not None else 7
        if matriz_empirica is not None and dim_n > 1500:
            st.info(f"💡 **Protección Big Data Activada:** Tu matriz tiene {n_sujetos:,} sujetos. Para evitar colapsar la RAM del servidor y dado que el Error Estándar ya tendió a cero, limitamos N=1500 para la simulación.")
            dim_n = 1500

        col_dim1, col_dim2 = st.columns(2)
        if matriz_empirica is None:
            with col_dim1: dim_n = st.number_input("Sujetos (n)", 5, 2000, 50, 5)
            with col_dim2: dim_m = st.number_input("Jueces (m)", 2, 30, 7, 1)
        
        escalera_no = None
        with st.spinner("Calculando límites del hiperespacio..."):
            if "Intervalar" in topologia: azar_N, min_N = ni_core.calcular_azar_termodinamico_ni(dim_m, k_escala), 0.0 
            elif "Nominal" in topologia:
                azar_N = nn_core.calcular_azar_termodinamico_nn(dim_m, k_escala)
                min_N = np.sqrt(min(nn_core._build_macrostate_dictionary_nn(dim_m, k_escala).keys())) 
            else: 
                _, escalera_no, _, min_N = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala)
                azar_N = np.sqrt(1.0 / k_escala)

        if "Ordinal" in topologia:
            st.info("🚧 **En Desarrollo:** El cálculo termodinámico exacto de las Pruebas de Cobertura y el Límite Físico para datos Ordinales se encuentra actualmente en fase de investigación y ajuste.")
        else:
            st.info(f"📏 **Límite Físico Inferior:** Geométricamente imposible obtener un acuerdo absoluto menor a **{min_N:.4f}**.")
        target_N = st.slider("Selecciona el Nivel de Consenso (N) objetivo:", float(min_N), 1.0, float(azar_N), 0.01, format="%.4f")
        
        if "Intervalar" in topologia: perc = ni_core.calcular_percentil_universal_ni(target_N, dim_m, k_escala)
        elif "Nominal" in topologia: perc = nn_core.calcular_percentil_universal_nn(target_N, dim_m, k_escala)
        else: _, _, perc, _ = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala, valor_observado=target_N)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mínimo Físico", f"{min_N:.4f}"); c2.metric("Azar Esperado", f"{azar_N:.4f}"); c3.metric("🏆 Percentil Objetivo", f"{perc:.2f} %")
        
        if "Ordinal" in topologia and escalera_no:
            st.dataframe(pd.DataFrame([escalera_no], index=["Consenso Mínimo Requerido"]).style.format("{:.4f}"), use_container_width=True)
        
        n_experimentos = st.number_input("Matrices a simular", 10, 500, 50, 10)
        carga = dim_n * dim_m * n_experimentos
        if carga > 1000000: st.error(f"🚨 **Límite excedido:** {carga:,} celdas. Reduce los experimentos.")
        
        if st.button("🔬 Iniciar Prueba", type="primary", disabled=(carga > 1000000)):
            with st.spinner("Simulando..."):
                res_cob = ejecutar_auditoria_cobertura(topologia, dim_n, dim_m, k_escala, target_N, n_experimentos, replicas)
                if res_cob:
                    df_cob = pd.DataFrame(res_cob)
                    ancho_ctrl = float(df_cob.iloc[0]["Ancho Medio IC"]) 
                    def alert_w(r):
                        try: return ['background: linear-gradient(90deg, #ffeeba, #f5c6cb); color: #721c24; font-weight: bold;']*len(r) if float(r['Cob. Población (%)']) >= 90.0 and float(r['Ancho Medio IC']) > (ancho_ctrl * 2.0) else ['']*len(r)
                        except: return ['']*len(r)
                    st.dataframe(df_cob.style.format({"Cob. Población (%)": "{:.1f}", "Cob. Muestra (%)": "{:.1f}", "µ(Población Real)": "{:.4f}", "µ(Valor Muestra)": "{:.4f}", "Ancho Medio IC": "{:.4f}"}).apply(alert_w, axis=1), use_container_width=True)

# ==============================================================================
# PESTAÑA 3: INVARIANZA TOPOLÓGICA
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🔄 Invarianza Topológica':
    st.markdown("### Auditoría de Invarianza (Sensibilidad Muestral)")
    if matriz_empirica is None or k_escala is None: 
        st.info("Sube una matriz y define k en el panel lateral.")
    else:
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        with c1: factor = st.number_input("Multiplicador (X veces):", 2, 200, 10)
        with c2: st.info(f"De **{n_sujetos}** a **{n_sujetos * factor}** sujetos.")
        
        if st.button("🔄 Ejecutar Test", type="primary"):
            m_rep = np.tile(matriz_empirica, (factor, 1))
            res_inv = []
            
            # Compresión Termodinámica al vuelo para el Marco N
            dict_original = Counter(tuple(x) for x in matriz_empirica)
            dict_replicada = Counter(tuple(x) for x in m_rep)
            
            with st.spinner("Calculando huella topológica..."):
                if "Intervalar" in topologia:
                    try:
                        v_ni_o = ni_core.calcular_estadisticas_ni_unificada(dict_original, k_escala, replicas)[0]
                        v_ni_r = ni_core.calcular_estadisticas_ni_unificada(dict_replicada, k_escala, replicas)[0]
                        res_inv.append({"Estimador": "N Interval (NI)", "Original": v_ni_o, "Replicada": v_ni_r, "Tipo": "N"})
                    except Exception as e: st.error(f"Error Marco N: {e}")
                    
                    try:
                        res_inv.append({"Estimador": "Alpha Krippendorff (AKI)", "Original": aki_core.calcular_estadisticas_aki(matriz_empirica, 10)['AKI Muestra'], "Replicada": aki_core.calcular_estadisticas_aki(m_rep, 10)['AKI Muestra'], "Tipo": "C"})
                        res_inv.append({"Estimador": "ICC(2,1)", "Original": icc21_core.calcular_estadisticas_icc21(matriz_empirica)['ICC Muestra'], "Replicada": icc21_core.calcular_estadisticas_icc21(m_rep)['ICC Muestra'], "Tipo": "C"})
                    except: pass
                    
                elif "Nominal" in topologia:
                    try:
                        v_nn_o = nn_core.calcular_estadisticas_nn_unificada(dict_original, k_escala, replicas)[0]
                        v_nn_r = nn_core.calcular_estadisticas_nn_unificada(dict_replicada, k_escala, replicas)[0]
                        res_inv.append({"Estimador": "N Nominal (NN)", "Original": v_nn_o, "Replicada": v_nn_r, "Tipo": "N"})
                    except Exception as e: st.error(f"Error Marco N: {e}")
                    
                    try:
                        res_inv.append({"Estimador": "Alpha Krippendorff (AKN)", "Original": akn_core.calcular_estadisticas_akn(matriz_empirica, 10, k_escala)['AKN Muestra'], "Replicada": akn_core.calcular_estadisticas_akn(m_rep, 10, k_escala)['AKN Muestra'], "Tipo": "C"})
                        res_inv.append({"Estimador": "Kappa Fleiss (KF)", "Original": kf_core.calcular_estadisticas_kf(matriz_empirica, 10, k_escala)['KF Muestra'], "Replicada": kf_core.calcular_estadisticas_kf(m_rep, 10, k_escala)['KF Muestra'], "Tipo": "C"})
                    except: pass
                    
                else: # Ordinal
                    try:
                        v_no_o = no_core.calcular_estadisticas_no_unificada(dict_original, k_escala, replicas)[0]
                        v_no_r = no_core.calcular_estadisticas_no_unificada(dict_replicada, k_escala, replicas)[0]
                        res_inv.append({"Estimador": "N Ordinal (NO)", "Original": v_no_o, "Replicada": v_no_r, "Tipo": "N"})
                    except Exception as e: st.error(f"Error Marco N: {e}")
                    
                    try:
                        res_inv.append({"Estimador": "Alpha Krippendorff (AKO)", "Original": ako_core.calcular_estadisticas_ako(matriz_empirica, 10, k_escala)['AKO Muestra'], "Replicada": ako_core.calcular_estadisticas_ako(m_rep, 10, k_escala)['AKO Muestra'], "Tipo": "C"})
                        res_inv.append({"Estimador": "Kendall W", "Original": w_core.calcular_estadisticas_w(matriz_empirica, 10, k_escala)['W Muestra'], "Replicada": w_core.calcular_estadisticas_w(m_rep, 10, k_escala)['W Muestra'], "Tipo": "C"})
                    except: pass

            df_inv = pd.DataFrame(res_inv)
            if not df_inv.empty:
                df_inv["Variación"] = df_inv["Replicada"] - df_inv["Original"]
                
                def style_inv(r):
                    try: 
                        return ['color: #28a745; font-weight: bold;' if abs(float(r['Variación'])) < 1e-6 else 'color: #dc3545; font-weight: bold; background-color: #ffe6e6;']*len(r)
                    except: 
                        return ['']*len(r)
                        
                st.dataframe(df_inv.drop(columns=["Tipo"]).style.format({"Original": "{:.4f}", "Replicada": "{:.4f}", "Variación": "{:+.4f}"}).apply(style_inv, axis=1), use_container_width=True)
                
                st.markdown("### Resumen Ejecutivo: Estabilidad Estructural")
                cols_inv = st.columns(len(df_inv))
                for i, row in df_inv.iterrows():
                    est = row["Estimador"]
                    rep = row["Replicada"]
                    dif = row["Variación"]
                    
                    if abs(dif) < 1e-6: 
                        cols_inv[i].metric(label=f"{est}", value=f"{rep:.4f}", delta="Invariante (0.0000)", delta_color="normal")
                    else: 
                        cols_inv[i].metric(label=f"{est}", value=f"{rep:.4f}", delta=f"Inconsistente ({dif:+.4f})", delta_color="inverse")

# ==============================================================================
# PESTAÑA 4: GENERADOR DE MATRICES EXPERIMENTALES (SINTÉTICAS)
# ==============================================================================
elif st.session_state["pestaña_activa"] == '🏗️ Generador de Matrices':
    st.subheader("🏗️ Generador de Matrices Sintéticas")
    st.error("###### ⚠️ Si tiene cargada una matriz, bórrela antes de proceder.")
    
    # 1. Configuración Base
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1: gen_k = st.number_input("Escala (k)", 2, 20, 5, key="gk")
    with g_col2: gen_m = st.number_input("Jueces (m)", 2, 50, 7, key="gm")
    with g_col3: gen_n = st.number_input("Sujetos (n)", 1, 50000, 15, key="gn")

    # 2. Configuración de Ruido
    g_col4, g_col5 = st.columns(2)
    with g_col4: gen_faltantes = st.slider("Faltantes (%)", 0.0, 100.0, 0.0, 5.0) / 100
    with g_col5: gen_decimales = st.slider("Decimales (Ruido)", 0, 2, 0)

    # 3. Patrón Base (Norma)
    st.markdown("#### Patrón Base (Norma)")
    if "df_base_1" not in st.session_state or st.session_state.get("gk_last_base1") != gen_k:
        st.session_state["df_base_1"] = pd.DataFrame(np.full((1, gen_k), round(100/gen_k, 2)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=["Prob (%)"])
        st.session_state["gk_last_base1"] = gen_k
    edit_base_1 = st.data_editor(st.session_state["df_base_1"], use_container_width=True, key="edit_base_1")

    # 4. Modo de Aplicación con Sincronización
    modo = st.radio("", ["Patrón global", "Patrón por sujeto"], horizontal=True)
    
    edit_n_rows = None
    if modo == "Patrón por sujeto":
        # Sincronización: Si cambió k, n o el patrón, reseteamos la tabla
        base_hash = hash(edit_base_1.values.tobytes())
        if st.session_state.get("last_state_key") != (gen_n, gen_k, base_hash):
            st.session_state["df_n_rows"] = pd.DataFrame(np.tile(edit_base_1.values, (gen_n, 1)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=[f"S{i+1}" for i in range(gen_n)])
            st.session_state["last_state_key"] = (gen_n, gen_k, base_hash)
        edit_n_rows = st.data_editor(st.session_state["df_n_rows"], use_container_width=True)

    # 5. Excepciones
    st.markdown("#### Excepciones (Agujeros Negros)")
    excepciones = st.multiselect("Selecciona sujetos para inyectar anomalías:", [f"S{i+1}" for i in range(gen_n)])
    edit_exc = None
    if excepciones:
        # Sincronización de excepciones
        if st.session_state.get("exc_last") != excepciones or st.session_state.get("exc_last_k") != gen_k:
            st.session_state["df_exc"] = pd.DataFrame(np.full((len(excepciones), gen_k), round(100/gen_k, 2)), columns=[f"V{i}" for i in range(1, gen_k+1)], index=excepciones)
            st.session_state["exc_last"] = excepciones
            st.session_state["exc_last_k"] = gen_k
        edit_exc = st.data_editor(st.session_state["df_exc"], use_container_width=True)

    def crear_matriz_sintetica():
        matriz = []
        base_p = edit_base_1.iloc[0].values
        
        for i in range(gen_n):
            sujeto_id = f"S{i+1}"
            if edit_exc is not None and sujeto_id in excepciones:
                p = edit_exc.loc[sujeto_id].values
            elif modo == "Patrón por sujeto" and edit_n_rows is not None:
                p = edit_n_rows.loc[sujeto_id].values
            else:
                p = base_p
            
            p_norm = p / p.sum() if p.sum() > 0 else np.ones(gen_k)/gen_k
            fila = np.random.choice(range(1, gen_k+1), size=gen_m, p=p_norm).astype(float)
            
            if gen_decimales > 0:
                fila += np.random.uniform(-0.1, 0.1, gen_m)
                fila = np.clip(fila, 1.0, float(gen_k))
                fila = np.round(fila, gen_decimales)
            if gen_faltantes > 0:
                fila[np.random.rand(gen_m) < gen_faltantes] = np.nan
            matriz.append(fila)
        return pd.DataFrame(matriz, columns=[f"J{i+1}" for i in range(gen_m)])

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Preparar Descarga", use_container_width=True):
            st.session_state['matriz_csv_temp'] = crear_matriz_sintetica()
        if 'matriz_csv_temp' in st.session_state:
            csv = st.session_state['matriz_csv_temp'].to_csv(index=False)
            st.download_button("Descargar CSV", csv, "matriz_sintetica.csv", "text/csv", use_container_width=True)

    with col_b2:
        if st.button("🚀 Generar y Cargar en la App", type="primary", use_container_width=True):
            with st.spinner("Sintetizando..."):
                df_sintetico = crear_matriz_sintetica()
                
                # Asumiendo que el procesamiento del DataFrame ya está integrado o
                # usas la función de cargar_y_agregar_dataset pasándole un buffer
                import io
                buffer = io.StringIO()
                df_sintetico.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                conteo_est, n_tot = cargar_y_agregar_dataset(buffer, sep=",")
                
                # Forzamos la redirección inmediata a Pestaña 1 e Intervalar
                st.session_state.update({
                    'usar_sintetica': True, 
                    'matriz_generada_app': df_sintetico, 
                    'df_original_raw': df_sintetico, 
                    'df': df_sintetico, 
                    'diccionario_estados': conteo_est, 
                    'n_total': n_tot, 
                    'k_escala': gen_k,
                    'topologia_activa': 'Intervalar (Continua)', 
                    'pestaña_activa': '📊 Cálculo del Consenso'      # <-- Fuerza la pestaña
                })
                st.rerun()

# ==============================================================================
# PESTAÑA 5: DUELO DE PRECISIÓN N vs todos los demás
# ==============================================================================
elif st.session_state["pestaña_activa"] == '⚔️ Duelo: N vs todos los demás':
    st.markdown("### ⚔️ Duelo de Precisión: Marco N vs todos los demás")
    if k_escala is None:
        st.warning("⚠️ Debes definir el límite del hiperespacio (k) en el panel lateral.")
    else:
        st.write("Esta simulación calcula el Error Cuadrático Medio (MSE) enfrentando la estimación de la muestra contra la Verdad Poblacional asintótica.")
        
        dim_n, dim_m = n_sujetos if matriz_empirica is not None else 50, m_jueces if matriz_empirica is not None else 7

        with st.spinner("Calculando límites del hiperespacio..."):
            if "Intervalar" in topologia:
                azar_N = ni_core.calcular_azar_termodinamico_ni(dim_m, k_escala)
                min_N = 0.0 
            elif "Nominal" in topologia:
                azar_N = nn_core.calcular_azar_termodinamico_nn(dim_m, k_escala)
                macro_dict = nn_core._build_macrostate_dictionary_nn(dim_m, k_escala)
                min_N = np.sqrt(min(macro_dict.keys())) 
            else: 
                _, _, _, min_N = no_core.analizar_termodinamica_no(dim_n, dim_m, k_escala)
                azar_N = np.sqrt(1.0 / k_escala)
        
        n_duelo = st.slider("Experimentos (Rondas de disparo)", 10, 200, 50)
        target_N = st.slider(
            "Consenso Objetivo (Tirador IA):", 
            min_value=float(min_N), max_value=1.0000, value=float(azar_N), step=0.0100, format="%.4f", key="slider_duelo"
        )
        
        if st.button("🔥 Iniciar Combate de Precisión", type="primary"):
            with st.spinner("Calculando trayectorias y calculando MSE..."):
                resultados = ejecutar_duelo_ia(topologia, dim_n, dim_m, k_escala, target_N, n_duelo, replicas)
                
                if resultados:
                    df_duel = pd.DataFrame(resultados)
                    st.markdown("### 📊 Resultado del Duelo")
                    
                    c1, c2, c3 = st.columns(3)
                    try:
                        n_mse = float(df_duel.iloc[0]["MSE (Error)"])
                        a_mse = float(df_duel.iloc[1]["MSE (Error)"])
                        n_cob = float(df_duel.iloc[0]["Cobertura (%)"].replace('%',''))
                        a_cob = float(df_duel.iloc[1]["Cobertura (%)"].replace('%',''))
                        
                        c1.metric("Fiabilidad N (Cob)", f"{n_cob}%", f"{n_cob - a_cob:.1f}% vs Alfa")
                        c2.metric("Precisión N (MSE)", f"{n_mse:.6f}", f"{n_mse - a_mse:.6f} vs Alfa", delta_color="inverse")
                    except:
                        pass
                        
                    st.dataframe(df_duel.style.highlight_min(subset=["MSE (Error)"], color="lightgreen"), use_container_width=True)

# ==============================================================================
# PESTAÑA 6: MANUAL DE USUARIO Y FUNDAMENTOS TEÓRICOS
# ==============================================================================
elif st.session_state["pestaña_activa"] == '📖 Manual de Usuario':
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.video("https://youtu.be/XSpIfelUrZU")
    with col3:
        st.write(""); st.write(""); st.write(""); st.write("")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.link_button(
            "📄 Leer N-Metrics Logic Blueprint (PDF)", 
            "https://docs.google.com/viewer?url=https://raw.githubusercontent.com/manuel-narsa/n-metrics/main/N-Metrics_Logic_Blueprint.pdf",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.video("https://youtu.be/f0LMSS5LEho")

    st.markdown("## 📖 Manual de Usuario y Fundamentos Teóricos")
    st.write("Bienvenido al entorno analítico de **Métricas N**. Esta plataforma permite evaluar el nivel de consenso real de matrices empíricas, superando las paradojas de los estimadores frecuentistas clásicos mediante la aplicación de la **Termodinámica Exacta de la Información**.")
    
    st.markdown("---")
    st.markdown("### 1. Carga de Datos y Topología (Panel Lateral)")
    st.write("El primer paso es suministrar la materia prima y definir las leyes físicas del hiperespacio de probabilidad.")
    
    st.markdown("""
    * **Archivo CSV:** Sube tu matriz empírica. El formato es estricto: **Sujetos en las filas ($n$)** y **Jueces/Evaluadores en las columnas ($m$)**. 
        * 🤖 *Autolimpiador Inteligente:* No te preocupes si tu Excel tiene cabeceras (ej. "Juez 1", "Evaluador A") o si la primera columna tiene los nombres de los sujetos (ej. "S001", "Paciente X"). El sistema lo detectará heurísticamente y recortará la matriz para procesar solo los datos puros.
    * **Naturaleza de los Datos (Topología):** Fundamental. Le dice al motor matemático cómo tratar las distancias.
        * *Intervalar:* Escalas numéricas continuas (ej. notas del 1 al 10).
        * *Ordinal:* Escalas categóricas con jerarquía estricta.
        * *Nominal:* Categóricas puras sin orden. **¡Novedad!** Soporta etiquetas de texto (ej. "Rojo", "Verde", "Enfermo", "Sano"). El sistema las mapeará internamente a clases topológicas (1, 2, 3...) de forma transparente.
    * **Categorías de la Escala máxima ($k$):** ¡Obligatorio! Es el valor máximo posible de tu escala. Define el "techo" y los límites físicos del hiperespacio.
    """)
    
    st.markdown("---")
    st.markdown("### 2. Generador de Matrices Sintéticas (Pestaña 4)")
    st.write("Crea matrices termodinámicas a medida para probar las capacidades del sistema o generar casos de estudio controlados.")
    st.markdown("""
    * **1. Patrón Base (Norma):** Define las probabilidades (pesos) de que un juez elija cada categoría.
    * **2. Aplicación del Patrón:** * *Global:* El patrón base aplica a toda la matriz por igual de forma rápida.
        * *Por Sujeto:* Abre una tabla interactiva para editar las probabilidades fila por fila, usando el Patrón Base como plantilla inicial (incluye botón de Sincronización).
    * **3. Excepciones (Agujeros Negros):** Aísla sujetos específicos y aplícales reglas de probabilidad totalmente caóticas o extremas para probar el escáner de anomalías.
    * **Integración Directa:** El botón **"Generar y Cargar en la App"** inyectará tu matriz sintética apagando temporalmente tu archivo CSV subido, permitiéndote evaluarla al instante.
    """)

    st.markdown("---")
    st.markdown("### 3. Inferencia de Consenso: Marco N vs Clásicos (Pestaña 1)")
    st.write("Esta sección somete tu matriz a una batalla de estimadores. El Marco N utiliza **Simulación Ponderada (SP)** para no anclarse al sesgo empírico.")
    st.markdown("""
    * **Consenso vs. Fiabilidad:** Los estimadores clásicos (como ICC o Alpha de Krippendorff) miden en realidad *Varianza*. Si tu matriz es muy homogénea (todos los sujetos son de sobresaliente y los jueces aciertan), estas métricas colapsarán y dirán que no hay fiabilidad. El **Marco N** mide verdadero *Consenso Geométrico*, manteniéndose estable aunque la varianza sea cero.
    * **El Suelo de Cristal:** Notarás que a veces el Intervalo de Confianza del Marco N *no contiene* a la Población Real y no baja de cierto número. Esto no es un error: es el simulador demostrándote que tu matriz empírica es tan anómala (ej. 14 perfectos y 1 catastrófico) que la proyección de su fiabilidad ha caído en una "zona prohibida" donde es matemáticamente imposible construir una matriz física.
    * **Percentil Universal:** Un 85% significa que tu matriz tiene más orden geométrico que el 85% de todos los universos caóticos posibles para tu diseño exacto ($n$, $m$, $k$).
    """)

    st.markdown("---")
    st.markdown("""
    ### 4. Auditoría Estructural: Escáner de Anomalías
    Identifica qué elementos de la matriz (sujetos problemáticos o jueces erráticos) están inyectando entropía excesiva en el sistema.

    El algoritmo calcula el **Acuerdo Local** de cada fila y lo compara con la termodinámica general de tu matriz. Marcará en rojo aquellos casos que caigan por debajo de un Límite Crítico dinámico, definido por el umbral de sensibilidad ($\sigma$) que tú elijas:
    """)
    st.latex(r"Límite = \mu_{local} - (\text{Umbral} \cdot \sigma_{local})")

    st.markdown("---")
    st.markdown("""
    ### 5. Pruebas de Estrés y Diagnóstico (Pestañas 2 y 3)
    Herramientas avanzadas para auditar la honestidad matemática de los estimadores:
    * **Auditoría de Cobertura (Stress Test):** Simula decenas de universos paralelos ensamblando tuplas combinatorias exactas. Demuestra cómo las métricas clásicas sufren de "Ceguera Espacial" y no logran capturar la Verdad Poblacional, mientras el Marco N mantiene una cobertura de seguridad.
    * **Auditoría de Invarianza:** El consenso debe ser una *Propiedad Intensiva* (como la temperatura: el agua hierve a 100ºC sea un vaso o un océano). Esta prueba clona tu matriz multiplicando artificialmente el número de sujetos ($n$). Verás cómo el Marco N se mantiene inalterable (Invariante), mientras que los estimadores clásicos bailan y cambian de valor arrastrados por los grados de libertad.
    """)
    st.markdown("---")
    st.markdown("""
    ### ⚔️ Guía del Duelo de Titanes (N vs. Krippendorff)
    
    Esta pestaña es un **laboratorio de estrés estadístico** diseñado para auditar la precisión de las métricas en condiciones críticas de Inteligencia Artificial. Aquí se enfrenta el **Marco Termodinámico N** contra el estándar actual, el **Alfa de Krippendorff (AK)**.
    
    #### 🧪 Descripción de los Experimentos
    
    1. **Paradoja de la Varianza Cero (Techo de Escala)**
       * **Objetivo:** Evaluar el comportamiento cuando existe un acuerdo casi total en una sola categoría.
       * **Dinámica:** Se genera una matriz donde los jueces coinciden masivamente. La estadística clásica suele colapsar (Bias alto) al no encontrar "varianza" que procesar.
       * **Dimensiones:** Sujetos ($N$) y Jueces ($M$) moderados, con un objetivo de concordancia ($target\_N$) > 0.9.
    
    2. **Estabilidad en Muestras Pequeñas (Expertos)**
       * **Objetivo:** Simular auditorías de alta especialización (médica/legal) con muy pocos evaluadores.
       * **Dinámica:** Se fuerza un escenario de $N < 10$ y $M$ entre 2 y 3. Demuestra si la métrica es capaz de extraer la "verdad" con recursos mínimos sin sesgarse.
       * **Dimensiones:** $N$ muy pequeño, $M$ mínimo, Escala $K$ completa.
    
    3. **Ruido y Dispersión (Entropía Máxima)**
       * **Objetivo:** Medir la capacidad de la métrica para identificar el caos.
       * **Dinámica:** Se genera una matriz de votos aleatorios o altamente discrepantes. N utiliza la entropía de configuración para penalizar el desorden de forma más quirúrgica.
       * **Dimensiones:** $N$ y $M$ estándar, $target\_N$ bajo (< 0.3).
    
    4. **Gran Escala (Stress Test de 1 Millón)**
       * **Objetivo:** Validar la consistencia asintótica para Big Data e IA.
       * **Dinámica:** Proyecta el comportamiento de la métrica hacia un millón de registros. Busca detectar si el error disminuye con el volumen o si la métrica arrastra un error sistemático (Bias).
       * **Dimensiones:** $N$ proyectado a $10^6$, $M$ variable.
    
    #### 📊 Glosario de Métricas de Auditoría
    
    Para demostrar la superioridad de un método sobre otro, observamos cuatro indicadores clave:
    
    * **Cobertura (%):** Porcentaje de veces que el intervalo de confianza atrapó la "Verdad Poblacional". El estándar científico de fiabilidad es el **95%**.
    * **MSE (Error Cuadrático Medio):** Es la métrica de precisión por excelencia en IA. Mide cuánto se aleja el valor estimado de la realidad. **Cuanto más bajo (cerca de 0), mejor.**
    * **Ancho IC (Eficiencia):** Indica la incertidumbre. Un intervalo más estrecho que mantiene una alta cobertura es señal de una métrica más potente y eficiente.
    * **Bias (Sesgo):** Indica si la métrica tiende a ser pesimista (negativo) u optimista (positivo). El valor ideal para una auditoría objetiva es **0.00**.
    """)

# ==============================================================================
# PESTAÑA 7: AUTORÍA Y LICENCIA
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
    st.write("**Licencia Abierta**. GNU GPLv3. Se concede plena libertad para utilizar, estudiar, modificar y distribuir este código fuente, bajo la estricta condición de que cualquier obra derivada se publique obligatoriamente bajo estos mismos términos, garantizando el acceso abierto al resto de la comunidad.")
    
    st.markdown("---")
    st.markdown("### 🎓 Cómo citar este trabajo")
    st.write("Si utilizas N-Metrics o el Marco Termodinámico N en tu investigación, por favor cita el manuscrito original:")
    
    st.info("Narbona-Sarria, M. (2026). N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus. Zenodo. https://doi.org/10.5281/zenodo.20075069")
    
    with st.expander("Ver formato BibTeX para LaTeX"):
        st.code("""@article{narbona_n_coefficient,
  title={N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus},
  author={Narbona Sarria, Manuel},
  journal={Preprint},
  year={2026},
  url={https://doi.org/10.5281/zenodo.20075069}
}""", language="bibtex")