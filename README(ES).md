# N-Framework: La Termodinámica Exacta del Consenso

**N-Framework** (paquete `n-metrics` v0.8.5) es un ecosistema analítico y computacional diseñado para superar las limitaciones estructurales de las métricas clásicas de acuerdo e inter-evaluador (tales como la Alpha de Krippendorff, el Kappa de Fleiss, el Coeficiente de Correlación Intraclase ICC(2,1) o la W de Kendall). Desarrollado por **Manuel Narbona Sarria**, el marco redefine la medida del consenso entre evaluadores reemplazando el análisis tradicional basado en varianza por una formulación exacta fundamentada en la **física estadística y la termodinámica del consenso geométrico**.

---

## 1. Fundamentos Teóricos y Cambio de Paradigma

### La Paradoja de las Métricas Clásicas
Las métricas clásicas de concordancia evalúan la fiabilidad basándose en la varianza entre sujetos. Esta premisa introduce fallos sistemáticos en escenarios de investigación reales:
1. **Paradoja de la Varianza Cero (Techo de Escala):** Cuando existe un acuerdo masivo o perfecto entre jueces en una misma categoría (homogeneidad muestral), la varianza se anula. En consecuencia, métricas como ICC o Krippendorff colapsan computacionalmente o entregan índices artificialmente bajos o indeterminados.
2. **Ceguera Espacial y Sensibilidad al Tamaño Muestral:** Los estimadores tradicionales varían en función del número de sujetos ($n$), tratando el acuerdo como una propiedad extensiva arrastrada por los grados de libertad.
3. **Sesgo Poblacional en Muestras Pequeñas:** En auditorías de alta especialización ($n < 10, m \in [2, 3]$), los métodos bootstrap clásicos sufren de sesgos optimistas o pesimistas severos.

### La Solución Termodinámica de N-Framework
N-Framework aísla las **firmas de consenso (macroestados)** de la matriz de evaluación $X \in \mathbb{R}^{n \times m}$, descomponiendo la estructura en el espacio de fases combinatorial:
* **Firma de Consenso (Macroestado):** Extrae la configuración topológica pura de respuestas por fila, descartando el ruido nominal de los sujetos y el orden de etiquetado.
* **Multiplicidad Teórica ($\Omega$):** Calcula mediante combinatoria exacta el volumen de microestados físicos que componen cada patrón en el espacio muestral.
* **Valor Poblacional Asintótico ($n \to \infty$):** Pondera las frecuencias de la muestra empírica con su volumen teórico en el universo de probabilidades, reconstruyendo de forma exacta el consenso verdadero libre de ruido estadístico.
* **Propiedad Intensiva:** El consenso bajo N-Framework se comporta como una variable intensiva (análoga a la temperatura), manteniéndose estrictamente **invariante ante la clonación o escalado de filas** ($n \to c \cdot n$).

---

## 2. Arquitectura del Sistema y Módulos del Motor

El núcleo del marco se organiza en una arquitectura modular limpia, optimizada para procesamiento de grandes volúmenes de datos, cómputo paralelo de caché de estados y una interfaz web interactiva en Streamlit (`app.py`).

```
n-metrics/
├── app.py                      # Aplicación Streamlit interactiva (9 Pestañas, i18n, GUI)
├── data_handler.py             # Ingesta streaming, autolimpiador de matrices y validación
├── calcular_diccionarios.py    # Generador paralelo de caché SQLite para macroestados (m, k)
├── pyproject.toml              # Configuración de empaquetado, dependencias y 18 comandos CLI
├── CITATION.cff                # Metadatos de citación académica
├── LICENSE                     # Licencia GNU General Public License v3.0
├── nmetrics/
│   ├── interval/
│   │   ├── ni_core.py          # Coeficiente N Intervalar (NI) y escáner de anomalías
│   │   ├── aki_core.py         # Comparador clásico: Adjusted Krippendorff Interval
│   │   └── icc21_core.py       # Comparador clásico: ICC(2,1) (F-ANOVA)
│   ├── nominal/
│   │   ├── nn_core.py          # Coeficiente N Nominal (NN) y particiones acotadas
│   │   ├── akn_core.py         # Comparador clásico: Adjusted Krippendorff Nominal
│   │   └── kf_core.py          # Comparador clásico: Fleiss' Kappa
│   └── ordinal/
│       ├── no_core.py          # Coeficiente N Ordinal (NO) y masa entrópica
│       ├── ako_core.py         # Comparador clásico: Adjusted Krippendorff Ordinal
│       └── w_core.py           # Comparador clásico: Kendall's W
```

### Componentes Clave del Motor

* **`app.py` (Orquestador Streamlit e i18n):** Integra un motor bilingüe de traducción dinámica (español/inglés) a través de `obtener_idioma_usuario()` y `t(key)`, gestión persistente de `st.session_state`, tablero de control lateral, y renderización de 9 módulos interactivos.
* **`data_handler.py` (Lector Big Data y Limpiador Heurístico):** Procesa archivos CSV/TXT mediante bloques (*chunksize = 100.000*) previniendo desbordamientos de memoria RAM. Su autolimpiador inteligente detecta cabeceras de texto y nombres de sujetos, recortando la matriz a datos numéricos puros y comprimiendo filas idénticas en tuplas de frecuencia mediante `collections.Counter`.
* **`calcular_diccionarios.py` (Generador de Caché SQLite):** Precalcula los diccionarios de macroestados para combinaciones de evaluadores ($m$) y categorías ($k$), almacenándolos en la base de datos `n_metrics_cache.db` (tabla `macroestados_cache`). Implementa ejecución en paralelo mediante `concurrent.futures.ProcessPoolExecutor` para $m < 60$ y procesamiento secuencial seguro con liberación explícita de memoria (`gc.collect()`) para $m \ge 60$.

---

## 3. Las Tres Topologías del Consenso

N-Framework adapta su motor matemático al tipo de variable y distancia métrica utilizada en el proceso de evaluación.

| Topología | Coeficiente N (Marco N) | Estimadores Clásicos de Control | Tipo de Datos y Reglas | Naturaleza de la Escala |
| :--- | :--- | :--- | :--- | :--- |
| **Intervalar** | **`NI`** | `AKI` (Krippendorff Interval), `ICC(2,1)` (F-ANOVA) | Escalas continuas o discretas numéricas ($k_{\text{min}}$ a $k_{\text{max}}$). | Distancias numéricas continuas donde las diferencias entre valores son constantes. |
| **Nominal** | **`NN`** | `AKN` (Krippendorff Nominal), `KF` (Fleiss Kappa) | Categórica pura. Mapea cadenas de texto a enteros $\{1, \dots, k\}$. Exige valores enteros. | Sin orden ni distancia implícita entre categorías. |
| **Ordinal** | **`NO`** | `AKO` (Krippendorff Ordinal), `W` (Kendall's W) | Categórica jerárquica. Exige valores enteros discretos. | Jerarquía estricta entre niveles con distancias de rango. |

---

## 4. Formulaciones y Mecánica Estadística

### 1. Extracción de Macroestados y Multiplicidad $\Omega$
Dada una matriz de evaluación con $n$ sujetos, $m$ jueces y $k$ niveles de escala, cada sujeto genera una distribución de frecuencias de asignación por categoría $\vec{f} = (f_1, f_2, \dots, f_k)$ tal que $\sum_{j=1}^k f_j = m$.

La multiplicidad física (número de microestados que generan el mismo macroestado) se calcula mediante la fórmula multinomial:

$$\Omega(\vec{f}) = \frac{m!}{f_1! f_2! \dots f_k!}$$

### 2. Azar Termodinámico ($P_e$) y Percentil Universal
El motor calcula la expectativa geométrica del azar $P_e$ para la configuración $(n, m, k)$. El **Percentil Universal (%)** evalúa la posición ordinal del consenso empírico dentro de la distribución combinatorial completa de universos posibles:

$$P_e = \text{Expectativa Geométrica bajo Azar Puro}(m, k)$$

Un Percentil Universal del 95% indica que la matriz empírica posee mayor orden geométrico que el 95% de todas las configuraciones estocásticas posibles para el diseño $(n, m, k)$.

### 3. Escáner de Anomalías y Suelo de Cristal
El escáner de anomalías compara la frecuencia empírica observada de cada macroestado ($P_{\text{emp}}$) frente a su probabilidad termodinámica teórica bajo azar puro ($P_{\text{azar}}$):

$$Z = \frac{\log_{10}(P_{\text{emp}} / P_{\text{azar}}) - \mu_{Z}}{\sigma_{Z}}$$

* **Sobrerrepresentación ($P_{\text{emp}} / P_{\text{azar}} > 1$):** Patrones de consenso observados con mayor frecuencia de la esperada por el azar.
* **Infrarrepresentación y Suelo de Cristal ($P_{\text{emp}} / P_{\text{azar}} < 1$):** Detecta sujetos o respuestas que caen en zonas de "suelo de cristal" (agujeros negros termodinámicos), donde la alta entropía o falta de grados de libertad imposibilita la proyección física estándar.

---

## 5. Módulos de la Aplicación Web (`app.py`)

La interfaz de usuario implementada en `app.py` contiene 9 pestañas especializadas:

1. **Pestaña 1: Inferencia de Consenso y Anomalías:** Carga de matrices empíricas, cálculo paralelo del Coeficiente N (`NI`, `NN`, `NO`) y métricas clásicas, simulación de Intervalos de Confianza ponderados, cálculo de $P_e$, percentiles universales y escáner de anomalías por $Z$-score.
2. **Pestaña 2: Pruebas de Cobertura:** Auditoría de stress testing para comprobar la tasa efectiva con la que los Intervalos de Confianza al 95% capturan la Verdad Poblacional en universos paralelos simulados.
3. **Pestaña 3: Informe Multirrango:** Explorador interactivo que analiza el comportamiento del consenso a lo largo de variaciones de escala $k$ y número de evaluadores $m$.
4. **Pestaña 4: Invarianza y Sensibilidad:** Demostración experimental de la propiedad intensiva del Marco N mediante el duplicado de filas ($n$), mostrando la estabilidad del Coeficiente N frente a la deriva de las métricas clásicas.
5. **Pestaña 5: Generador de Matrices Sintéticas:** Herramienta de simulación para crear matrices sintéticas especificando patrones de probabilidad global o por sujeto, e inyectando excepciones entrópicas ("agujeros negros") para pruebas de diagnóstico.
6. **Pestaña 6: Duelo de Precisión (N vs. Clásicos):** Suite de benchmarking que enfrenta a N-Framework contra métricas tradicionales en tres escenarios límite:
   * *Paradoja de Varianza Cero:* Evaluación bajo concordancia masiva ($target\_N > 0.9$).
   * *Muestras Pequeñas (Expertos):* Auditorías con $n < 10$ y $m \in [2, 3]$.
   * *Ruido y Entropía Máxima:* Dispersión caótica.
7. **Pestaña 7: Base de Datos de Diccionarios y Firmas:** Explorador e inspector de la base de datos SQLite (`n_metrics_cache.db`), permitiendo auditar firmas precalculadas y exportar diccionarios de macroestados en JSON.
8. **Pestaña 8: Manual de Usuario y Fundamentos Teóricos:** Guía completa e interactiva con fundamentación matemática, glosario y documentación bilingüe.
9. **Pestaña 9: Autoría y Licencia:** Créditos de desarrollo (Manuel Narbona Sarria), información de versión (0.8.5), cita en formato `CITATION.cff` y términos de la licencia GNU GPL v3.0.

---

## 6. Instalación, Requisitos y CLI

### Requisitos del Sistema
* **Python:** `>= 3.8`
* **Dependencias Principales:** `numpy >= 1.20.0`, `pandas >= 1.3.0`, `scipy >= 1.7.0`, `streamlit`, `plotly`, `Pillow`.

### Instalación
```bash
# Instalación desde PyPI
pip install n-metrics

# O instalación local desde el repositorio clonado
pip install -e .
```

### Comandos de la Interfaz de Línea de Comandos (CLI)

El archivo `pyproject.toml` registra 18 comandos ejecutables directamente desde la terminal:

```bash
# --- Tablas Comparativas Globales ---
nmetrics-interval   # Análisis comparativo para topología intervalar
nmetrics-nominal    # Análisis comparativo para topología nominal
nmetrics-ordinal    # Análisis comparativo para topología ordinal

# --- Métricas Individuales de N-Framework ---
nmetrics-ni         # Análisis individual NI (Intervalar)
nmetrics-nn         # Análisis individual NN (Nominal)
nmetrics-no         # Análisis individual NO (Ordinal)

# --- Métricas Individuales Clásicas ---
nmetrics-aki        # Análisis AKI (Krippendorff Interval)
nmetrics-icc21      # Análisis ICC(2,1)
nmetrics-akn        # Análisis AKN (Krippendorff Nominal)
nmetrics-kf         # Análisis Fleiss' Kappa
nmetrics-ako        # Análisis AKO (Krippendorff Ordinal)
nmetrics-w          # Análisis Kendall's W

# --- Pruebas de Cobertura y Estrés ---
nmetrics-ni-cob     # Cobertura NI
nmetrics-nn-cob     # Cobertura NN
nmetrics-no-cob     # Cobertura NO

# --- Módulos Comparativos de Resampling ---
nmetrics-interval-comp
nmetrics-nominal-comp
nmetrics-ordinal-comp
```

---

## 7. Citación y Licencia

### Citación Académica
Si utiliza N-Framework (`n-metrics`) en sus investigaciones o desarrollos académicos, por favor cite el trabajo según los metadatos de `CITATION.cff`:

```bibtex
@software{Narbona_Sarria_N-Metrics_The_Exact_2026,
  author = {Narbona Sarria, Manuel},
  title = {{N-Metrics: The Exact Thermodynamics of Consensus}},
  version = {0.8.5},
  year = {2026},
  url = {https://github.com/manuel-narsa/n-metrics}
}
```

### Licencia
Este proyecto está distribuido bajo la licencia **GNU General Public License v3.0 (GPLv3)**. Para más detalles, consulte el archivo `LICENSE.txt`.
