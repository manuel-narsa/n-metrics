import streamlit as st
import pandas as pd
import numpy as np
import time

# --- IMPORTACIONES DEL NÚCLEO DE N-METRICS ---
try:
    from nmetrics.interval import ni_core, aki_core, icc21_core
    from nmetrics.nominal import nn_core, akn_core, kf_core
    from nmetrics.ordinal import no_core, ako_core, w_core
    from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica
    LIBRERIA_CARGADA = True
except ImportError:
    LIBRERIA_CARGADA = False

# --- FUNCIÓN UNIFICADA DE AUDITORÍA DE COBERTURA ---
def ejecutar_auditoria_cobertura(topologia, n_sujetos, m_jueces, k_escala, escenario, n_experimentos, replicas):
    resultados = []
    
    if "Intervalar" in topologia:
        tipo_escala, decimales = 'paramétrica', 1
        modelos = ['NI (SP)', 'AKI (BC)', 'ICC(2,1) [ANOVA]']
    elif "Nominal" in topologia:
        tipo_escala, decimales = 'categórica', 0
        modelos = ['NN (SP)', 'AKN (BC)', 'KF (BC)']
    else:
        tipo_escala, decimales = 'ordinal', 0
        modelos = ['NO (SP)', 'AKO (BC)', 'Kendall W (BC)']

    data = {m: {'cobs_pob': [], 'cobs_mue': []} for m in modelos}
    barra = st.progress(0)
    
    for i in range(n_experimentos):
        matriz = generar_matriz_dinamica(n_sujetos, m_jueces, k_escala, escenario, tipo_escala=tipo_escala, decimales=decimales)
        
        # --- LÓGICA INTERVALAR ---
        if "Intervalar" in topologia:
            # NI
            v_m, v_p, ic_i, ic_s, *_ = ni_core.calcular_estadisticas_ni(matriz, replicas, k_escala, 'SP')
            if not np.isnan(v_m) and not np.isnan(ic_i):
                data[modelos[0]]['cobs_pob'].append(1 if (ic_i <= v_p <= ic_s) else 0)
                data[modelos[0]]['cobs_mue'].append(1 if (ic_i <= v_m <= ic_s) else 0)
            
            # AKI
            p_aki = aki_core.calcular_aki_poblacion_asintotica(matriz, 1, k_escala, 100)
            st_aki = aki_core.calcular_estadisticas_aki(matriz, replicas)
            if not np.isnan(st_aki['AKI Muestra']) and not np.isnan(st_aki['IC Inf']):
                data[modelos[1]]['cobs_pob'].append(1 if (st_aki['IC Inf'] <= p_aki <= st_aki['IC Sup']) else 0)
                data[modelos[1]]['cobs_mue'].append(1 if (st_aki['IC Inf'] <= st_aki['AKI Muestra'] <= st_aki['IC Sup']) else 0)
                
            # ICC
            p_icc = icc21_core.calcular_icc_poblacion_asintotica(matriz, 1, k_escala, 100)
            st_icc = icc21_core.calcular_estadisticas_icc21(matriz)
            if not np.isnan(st_icc['ICC Muestra']) and not np.isnan(st_icc['IC Inf']):
                data[modelos[2]]['cobs_pob'].append(1 if (st_icc['IC Inf'] <= p_icc <= st_icc['IC Sup']) else 0)
                data[modelos[2]]['cobs_mue'].append(1 if (st_icc['IC Inf'] <= st_icc['ICC Muestra'] <= st_icc['IC Sup']) else 0)

        # --- LÓGICA NOMINAL ---
        elif "Nominal" in topologia:
            # NN
            v_m, v_p, ic_i, ic_s, *_ = nn_core.calcular_estadisticas_nn(matriz, replicas, k_escala, 'SP')
            if not np.isnan(v_m) and not np.isnan(ic_i):
                data[modelos[0]]['cobs_pob'].append(1 if (ic_i <= v_p <= ic_s) else 0)
                data[modelos[0]]['cobs_mue'].append(1 if (ic_i <= v_m <= ic_s) else 0)
                
            # AKN
            p_akn = akn_core.calcular_akn_poblacion_asintotica(matriz, k_escala, 100)
            st_akn = akn_core.calcular_estadisticas_akn(matriz, replicas)
            if not np.isnan(st_akn['AKN Muestra']):
                data[modelos[1]]['cobs_pob'].append(1 if (st_akn['IC Inf'] <= p_akn <= st_akn['IC Sup']) else 0)
                data[modelos[1]]['cobs_mue'].append(1 if (st_akn['IC Inf'] <= st_akn['AKN Muestra'] <= st_akn['IC Sup']) else 0)
                
            # KF
            p_kf = kf_core.calcular_kf_poblacion_asintotica(matriz, k_escala, 100)
            st_kf = kf_core.calcular_estadisticas_kf(matriz, replicas)
            if not np.isnan(st_kf['KF Muestra']):
                data[modelos[2]]['cobs_pob'].append(1 if (st_kf['IC Inf'] <= p_kf <= st_kf['IC Sup']) else 0)
                data[modelos[2]]['cobs_mue'].append(1 if (st_kf['IC Inf'] <= st_kf['KF Muestra'] <= st_kf['IC Sup']) else 0)

        # --- LÓGICA ORDINAL ---
        else:
            # NO
            v_m, v_p, ic_i, ic_s, *_ = no_core.calcular_estadisticas_no(matriz, replicas, k_escala, 'SP')
            if not np.isnan(v_m) and not np.isnan(ic_i):
                data[modelos[0]]['cobs_pob'].append(1 if (ic_i <= v_p <= ic_s) else 0)
                data[modelos[0]]['cobs_mue'].append(1 if (ic_i <= v_m <= ic_s) else 0)
                
            # AKO
            p_ako = ako_core.calcular_ako_poblacion_asintotica(matriz, k_escala, 100)
            st_ako = ako_core.calcular_estadisticas_ako(matriz, replicas)
            if not np.isnan(st_ako['AKO Muestra']):
                data[modelos[1]]['cobs_pob'].append(1 if (st_ako['IC Inf'] <= p_ako <= st_ako['IC Sup']) else 0)
                data[modelos[1]]['cobs_mue'].append(1 if (st_ako['IC Inf'] <= st_ako['AKO Muestra'] <= st_ako['IC Sup']) else 0)
                
            # W
            p_w = w_core.calcular_w_poblacion_asintotica(matriz, k_escala, 100)
            st_w = w_core.calcular_estadisticas_w(matriz, replicas)
            if not np.isnan(st_w['W Muestra']):
                data[modelos[2]]['cobs_pob'].append(1 if (st_w['IC Inf'] <= p_w <= st_w['IC Sup']) else 0)
                data[modelos[2]]['cobs_mue'].append(1 if (st_w['IC Inf'] <= st_w['W Muestra'] <= st_w['IC Sup']) else 0)

        barra.progress((i + 1) / n_experimentos)

    for m in modelos:
        resultados.append({
            "Estimador": m,
            "Cobertura Población (%)": np.mean(data[m]['cobs_pob']) * 100 if data[m]['cobs_pob'] else 0.0,
            "Cobertura Muestra (%)": np.mean(data[m]['cobs_mue']) * 100 if data[m]['cobs_mue'] else 0.0
        })
        
    return resultados, modelos

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Métricas N: Marco Termodinámico", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .metric-box { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Métricas N: La Termodinámica Exacta del Consenso")
st.markdown("Plataforma oficial para la inferencia termodinámica y auditoría topológica de matrices empíricas.")

if not LIBRERIA_CARGADA:
    st.error("⚠️ La librería 'n-metrics' no está instalada. Ejecuta 'pip install n-metrics'.")
    st.stop()

# --- PANEL LATERAL ---
st.sidebar.header("📂 Carga de Datos (Opcional para Pruebas)")
archivo_subido = st.sidebar.file_uploader("Sube tu matriz empírica (CSV)", type=["csv"])
separador = st.sidebar.selectbox("Separador del CSV", [",", ";", "\t"])

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Configuración Topológica")
topologia = st.sidebar.radio("Naturaleza de los datos:", ["Intervalar (Continua)", "Nominal (Categórica sin orden)", "Ordinal (Categórica ordenada)"])

# Dejamos la caja vacía sin valor por defecto
k_escala = st.sidebar.number_input("Categorías de la Escala máxima (k)", min_value=2, value=None, placeholder="Ej: 5", step=1)
replicas = st.sidebar.slider("Réplicas de Simulación (S)", min_value=100, max_value=5000, value=1000, step=100)

# --- PROCESAMIENTO INICIAL Y BLINDAJE ESTRICTO DE ESCALA ---
df = None
matriz_empirica = None
n_sujetos_defecto = 50
m_jueces_defecto = 7

if archivo_subido:
    try:
        df = pd.read_csv(archivo_subido, sep=separador, header=None)
        df.columns = [f"Juez {i+1}" for i in range(df.shape[1])]
        n_sujetos, m_jueces = df.shape
        df.index = [f"S{str(i+1).zfill(3)}" for i in range(n_sujetos)]
        matriz_empirica = df.values
        
        valor_maximo_detectado = np.nanmax(matriz_empirica)
        
        # 1. Si no ha introducido nada, pedimos el dato y detenemos la app
        if k_escala is None:
            st.sidebar.info(f"⚠️ Por favor, introduce el valor de la escala (k) arriba. La nota empírica más alta detectada en tu matriz es un **{valor_maximo_detectado:.0f}**.")
            st.stop()
            
        # 2. Si ha introducido un dato imposible, bloqueamos y damos error
        elif k_escala < valor_maximo_detectado:
            st.sidebar.error(f"🚨 **¡Error Topológico!** Has indicado una escala máxima de k={k_escala}, pero tu matriz contiene puntuaciones de hasta **{valor_maximo_detectado:.0f}**. El valor de 'k' debe ser mayor o igual a {valor_maximo_detectado:.0f}.")
            st.stop()
            
        # 3. Si todo está correcto, damos luz verde
        else:
            st.sidebar.success(f"✅ Escala validada geométricamente (k={k_escala}).")
            
    except Exception as e:
        st.sidebar.error(f"Error al leer el CSV: {e}")

# --- PESTAÑAS (Aparecen siempre) ---
tab_inferencia, tab_cobertura, tab_autoria = st.tabs(["📊 Inferencia de Consenso", "🎯 Pruebas de Cobertura", "📜 Autoría y Licencia"])

# PESTAÑA 1: INFERENCIA 
with tab_inferencia:
    if matriz_empirica is None:
        st.info("👆 Sube un archivo CSV en el panel lateral izquierdo para calcular la inferencia de tu matriz empírica.")
    else:
        with st.expander("👁️ Vista previa de la matriz empírica", expanded=False):
            st.write(f"**Dimensiones:** {n_sujetos} sujetos × {m_jueces} jueces.")
            st.dataframe(df, height=300, use_container_width=True)

        st.markdown("### Batalla de Estimadores")
        if "Intervalar" in topologia: opciones = ["NI (Marco N)", "AKI (Bootstrap C.)", "ICC(2,1) (F-ANOVA)"]
        elif "Nominal" in topologia: opciones = ["NN (Marco N)", "AKN (Bootstrap C.)", "Kappa Fleiss (Bootstrap C.)"]
        else: opciones = ["NO (Marco N)", "AKO (Bootstrap C.)", "Kendall W (Bootstrap C.)"]
            
        estimadores = st.multiselect("Selecciona los modelos a ejecutar:", opciones, default=[opciones[0]])

        if st.button("🚀 Calcular Intervalos", type="primary"):
            if not estimadores: 
                st.warning("Selecciona al menos un estimador.")
            else:
                resultados_inf = []
                barra_inf = st.progress(0)
                t_inicio = time.time()

                with st.spinner("Explorando el hiperespacio termodinámico..."):
                    # TOPOLOGÍA INTERVALAR
                    if "Intervalar" in topologia:
                        if "NI (Marco N)" in estimadores:
                            res_ni = ni_core.calcular_estadisticas_ni(matriz_empirica, replicas, k_escala, 'SP')
                            v_mu, v_pob, ic_i, ic_s = res_ni[0], res_ni[1], res_ni[2], res_ni[3]
                            
                            # Llamadas directas al motor termodinámico
                            p_e = ni_core.calcular_azar_termodinamico_ni(m_jueces, k_escala)
                            percentil = ni_core.calcular_percentil_universal_ni(v_pob, m_jueces, k_escala)
                            
                            resultados_inf.append({"Métrica": "N Interval (NI)", "Muestra": v_mu, "Pob. Real": v_pob, "IC Inf": ic_i, "IC Sup": ic_s, "Valor Azar": p_e, "Percentil (%)": percentil, "Motor": "SP"})
                        
                        if "AKI (Bootstrap C.)" in estimadores:
                            pob_aki = aki_core.calcular_aki_poblacion_asintotica(matriz_empirica, 1, k_escala, 100)
                            stats = aki_core.calcular_estadisticas_aki(matriz_empirica, replicas)
                            resultados_inf.append({"Métrica": "Alpha Krippendorff (AKI)", "Muestra": stats['AKI Muestra'], "Pob. Real": pob_aki, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "Bootstrap"})
                        
                        if "ICC(2,1) (F-ANOVA)" in estimadores:
                            pob_icc = icc21_core.calcular_icc_poblacion_asintotica(matriz_empirica, 1, k_escala, 100)
                            stats = icc21_core.calcular_estadisticas_icc21(matriz_empirica)
                            resultados_inf.append({"Métrica": "ICC(2,1)", "Muestra": stats['ICC Muestra'], "Pob. Real": pob_icc, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "F-ANOVA"})
                    
                    # TOPOLOGÍA NOMINAL
                    elif "Nominal" in topologia:
                        if "NN (Marco N)" in estimadores:
                            res_nn = nn_core.calcular_estadisticas_nn(matriz_empirica, replicas, k_escala, 'SP')
                            v_mu, v_pob, ic_i, ic_s = res_nn[0], res_nn[1], res_nn[2], res_nn[3]
                            
                            p_e = nn_core.calcular_azar_termodinamico_nn(m_jueces, k_escala)
                            percentil = nn_core.calcular_percentil_universal_nn(v_pob, m_jueces, k_escala)
                            
                            resultados_inf.append({"Métrica": "N Nominal (NN)", "Muestra": v_mu, "Pob. Real": v_pob, "IC Inf": ic_i, "IC Sup": ic_s, "Valor Azar": p_e, "Percentil (%)": percentil, "Motor": "SP"})
                        
                        if "AKN (Bootstrap C.)" in estimadores:
                            pob_akn = akn_core.calcular_akn_poblacion_asintotica(matriz_empirica, k_escala, 100)
                            stats = akn_core.calcular_estadisticas_akn(matriz_empirica, replicas)
                            resultados_inf.append({"Métrica": "Alpha Krippendorff (AKN)", "Muestra": stats['AKN Muestra'], "Pob. Real": pob_akn, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "Bootstrap"})
                        
                        if "Kappa Fleiss (Bootstrap C.)" in estimadores:
                            pob_kf = kf_core.calcular_kf_poblacion_asintotica(matriz_empirica, k_escala, 100)
                            stats = kf_core.calcular_estadisticas_kf(matriz_empirica, replicas)
                            resultados_inf.append({"Métrica": "Kappa Fleiss (KF)", "Muestra": stats['KF Muestra'], "Pob. Real": pob_kf, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "Bootstrap"})

                    # TOPOLOGÍA ORDINAL
                    elif "Ordinal" in topologia:
                        if "NO (Marco N)" in estimadores:
                            res_no = no_core.calcular_estadisticas_no(matriz_empirica, replicas, k_escala, 'SP')
                            v_mu, v_pob, ic_i, ic_s = res_no[0], res_no[1], res_no[2], res_no[3]
                            
                            # Ordinal requiere n_sujetos y devuelve tuplas
                            p_e_tuple = no_core.calcular_azar_termodinamico_no_analitico_exacto(n_sujetos, m_jueces, k_escala)
                            p_e = p_e_tuple[0] 
                            
                            percentil_tuple = no_core.calcular_percentil_universal_no_exacto(v_pob, n_sujetos, m_jueces, k_escala)
                            percentil = percentil_tuple[0]
                            
                            resultados_inf.append({"Métrica": "N Ordinal (NO)", "Muestra": v_mu, "Pob. Real": v_pob, "IC Inf": ic_i, "IC Sup": ic_s, "Valor Azar": p_e, "Percentil (%)": percentil, "Motor": "SP"})
                        
                        if "AKO (Bootstrap C.)" in estimadores:
                            pob_ako = ako_core.calcular_ako_poblacion_asintotica(matriz_empirica, k_escala, 100)
                            stats = ako_core.calcular_estadisticas_ako(matriz_empirica, replicas)
                            resultados_inf.append({"Métrica": "Alpha Krippendorff (AKO)", "Muestra": stats['AKO Muestra'], "Pob. Real": pob_ako, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "Bootstrap"})
                        
                        if "Kendall W (Bootstrap C.)" in estimadores:
                            pob_w = w_core.calcular_w_poblacion_asintotica(matriz_empirica, k_escala, 100)
                            stats = w_core.calcular_estadisticas_w(matriz_empirica, replicas)
                            resultados_inf.append({"Métrica": "Kendall W", "Muestra": stats['W Muestra'], "Pob. Real": pob_w, "IC Inf": stats['IC Inf'], "IC Sup": stats['IC Sup'], "Valor Azar": np.nan, "Percentil (%)": np.nan, "Motor": "Bootstrap"})

                    barra_inf.progress(100)
                
                st.success(f"✅ Inferencia completada en {time.time() - t_inicio:.2f} segundos.")
                if resultados_inf:
                    df_res = pd.DataFrame(resultados_inf)
                    
                    # Formateador robusto
                    def format_4d(x):
                        if pd.isna(x): return "-"
                        try: return f"{float(x):.4f}"
                        except: return str(x)
                        
                    def format_2d(x):
                        if pd.isna(x): return "-"
                        try: return f"{float(x):.2f}"
                        except: return str(x)
                        
                    df_estilizado = df_res.style.format({
                        "Muestra": format_4d, 
                        "Pob. Real": format_4d, 
                        "IC Inf": format_4d, 
                        "IC Sup": format_4d,
                        "Valor Azar": format_4d,
                        "Percentil (%)": format_2d
                    })
                    
                    st.dataframe(df_estilizado, use_container_width=True)

# PESTAÑA 2: PRUEBAS DE COBERTURA
with tab_cobertura:
    st.markdown("### Auditoría de Paradoja de Cobertura (Stress Test Global)")
    
    # Determinar qué dimensiones usar
    if matriz_empirica is not None:
        dim_n = n_sujetos
        dim_m = m_jueces
        st.write(f"Utilizando las dimensiones de tu matriz empírica (**{dim_n} sujetos × {dim_m} jueces**), esta prueba genera universos paralelos para evaluar si los estimadores logran capturar la Verdad Termodinámica de la población.")
    else:
        st.write("No hay matriz cargada. Define las dimensiones para la simulación:")
        col_dim1, col_dim2 = st.columns(2)
        with col_dim1: dim_n = st.number_input("Sujetos (n)", min_value=5, value=50, step=5)
        with col_dim2: dim_m = st.number_input("Jueces (m)", min_value=2, value=7, step=1)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        n_experimentos = st.number_input("Número de matrices a simular", min_value=10, max_value=500, value=50, step=10)
    with col_c2:
        escenarios_disp = ["Casi Nulo", "Aleatorio", "Razonable", "Casi Perfecto", "Casi Idéntico"]
        escenario_sel = st.selectbox("Escenario de ruido (Densidad)", escenarios_disp, index=2)
        
    st.info("💡 **Recordatorio:** La métrica tradicional falla al capturar la Población porque ancla su intervalo rígidamente al sesgo empírico de la muestra. El Marco N usa corrección topológica.")

    if st.button("🔬 Iniciar Prueba de Estrés", type="primary"):
        with st.spinner(f"Simulando {n_experimentos} experimentos en hiperespacio {topologia.split(' ')[0]}..."):
            t_cob = time.time()
            
            resultados_cob, nombres_modelos = ejecutar_auditoria_cobertura(
                topologia, dim_n, dim_m, k_escala, escenario_sel, n_experimentos, replicas
            )
            
            st.success(f"✅ Stress Test finalizado en {time.time() - t_cob:.1f} segundos.")
            
            df_cob = pd.DataFrame(resultados_cob)
            st.dataframe(
                df_cob.style.format("{:.1f}", subset=["Cobertura Población (%)", "Cobertura Muestra (%)"])
                .background_gradient(cmap='Greens', subset=['Cobertura Población (%)']), 
                use_container_width=True
            )
            
            st.markdown("### Resumen Ejecutivo: Cobertura de la Población")
            
            # Tarjetas dinámicas (3 columnas) y comparación justa
            columnas_res = st.columns(len(df_cob))
            val_n = df_cob.iloc[0]["Cobertura Población (%)"] 
            
            for i, row in df_cob.iterrows():
                nombre_modelo = row["Estimador"]
                val_modelo = row["Cobertura Población (%)"]
                
                if i == 0:
                    columnas_res[i].metric(label=f"Marco N ({nombre_modelo})", value=f"{val_modelo:.1f} %", delta="Control Geométrico Absoluto")
                else:
                    diferencia = val_modelo - val_n
                    if diferencia < -5.0:
                        mensaje_delta = f"{diferencia:.1f} % (Ceguera Espacial)"
                        color_delta = "normal" 
                    else:
                        mensaje_delta = "Cobertura estable"
                        color_delta = "off"
                        
                    columnas_res[i].metric(label=f"Tradicional ({nombre_modelo})", value=f"{val_modelo:.1f} %", delta=mensaje_delta, delta_color=color_delta)

# PESTAÑA 3: AUDITORÍA Y LICENCIA
with tab_autoria:
    st.markdown("### Código Abierto y Transparencia")
    st.write("El Marco Termodinámico $N$ es un proyecto de ciencia abierta que resuelve las inconsistencias topológicas y las paradojas de acuerdo de los estimadores frecuentistas clásicos. Ha sido creado por Manuel Narbona Sarria")
    
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
    
    st.info("**Narbona Sarria, M. (202X).** *N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus*.")
    
    with st.expander("Ver formato BibTeX para LaTeX"):
        st.code("""@article{narbona_n_coefficient,
  title={N, the Natural Concordance Coefficient: The Exact Thermodynamics of Consensus},
  author={Narbona Sarria, Manuel},
  journal={Preprint},
  year={202X},
  url={URL_DE_TU_ARTICULO_O_GITHUB}
}""", language="bibtex")