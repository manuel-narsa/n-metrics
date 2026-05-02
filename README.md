

```
# 🚀 N-Metrics: The Exact Thermodynamics of Consensus

`nmetrics` es una librería avanzada de Python diseñada para evaluar la fiabilidad y el consenso en matrices de datos empíricos. Propone un nuevo marco topológico y termodinámico (El Marco N) que supera las limitaciones de la inferencia frecuentista asintótica, situando a la población en el centro del Intervalo de Confianza mediante IPW (Simulación Ponderada).

Además de los nuevos coeficientes Naturales (NI, NN, NO), la librería incluye motores ultra-optimizados (vectorizados y en C) para calcular los estimadores clásicos de consenso y sus remuestreos por Bootstrap Clásico, así como un avanzado motor de simulación de Monte Carlo para someter a los estimadores a pruebas de estrés.

---

## 🗺️ Cuadro General de la Librería

La arquitectura de `nmetrics` se divide en tres topologías fundamentales. Cada una cuenta con su Coeficiente Natural, estimadores clásicos de contraste y un arsenal de comandos de consola (CLI) para análisis empírico y simulaciones teóricas.

| Topología | Escala | Coeficiente Natural | Estimadores Clásicos | Comandos de Análisis CLI | Comandos de Stress Test (Cobertura) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intervalar** | Numérica Continua | **NI** (Natural Interval) | **AKI** (Alpha Krippendorff)<br>**ICC(2,1)** (F-ANOVA) | `nmetrics-ni`<br>`nmetrics-aki`<br>`nmetrics-icc21` | `nmetrics-ni-cob`<br>`nmetrics-aki-cob`<br>`nmetrics-icc21-cob` |
| **Nominal** | Categórica (Sin orden) | **NN** (Natural Nominal) | **AKN** (Alpha Krippendorff)<br>**KF** (Kappa Fleiss) | `nmetrics-nn`<br>`nmetrics-akn`<br>`nmetrics-kf` | `nmetrics-nn-cob`<br>`nmetrics-akn-cob`<br>`nmetrics-kf-cob` |
| **Ordinal** | Categórica (Ordenada) | **NO** (Natural Ordinal) | **AKO** (Alpha Krippendorff)<br>**W** (Kendall W) | `nmetrics-no`<br>`nmetrics-ako`<br>`nmetrics-w` | `nmetrics-no-cob`<br>`nmetrics-ako-cob`<br>`nmetrics-w-cob` |

*💡 **Comandos Globales:** Además de los individuales, puedes usar `nmetrics-interval`, `nmetrics-nominal` y `nmetrics-ordinal` para generar tablas comparativas de todos los coeficientes de una misma familia.*

---

## 🎲 Motor de Simulación y Generador de Escenarios

El núcleo de validación matemática de la librería reside en su módulo `generador_escenarios`. Este motor permite recrear paneles de jueces virtuales (Sujetos x Evaluadores) bajo condiciones controladas de ruido termodinámico y sesgo.

### ¿Qué hace el Generador de Escenarios?
Permite modelar el comportamiento de evaluación humana inyectando "ruido paramétrico" (para escalas intervalares) o "ruido categórico" (para escalas nominales/ordinales). Incluye 5 escenarios universales predefinidos:
1. **Casi Nulo:** Acuerdo destruido por entropía máxima (ruido extremo).
2. **Aleatorio:** Respuestas generadas por puro azar combinatorio.
3. **Razonable:** Consenso humano típico con desacuerdos locales moderados.
4. **Casi Perfecto:** Jueces altamente calibrados con desviaciones mínimas.
5. **Casi Idéntico:** Unanimidad absoluta o varianza topológica tendente a cero.

### Análisis de Cobertura (Stress Testing)
Gracias a los 9 comandos CLI terminados en `-cob` (ej. `nmetrics-ni-cob`), puedes cruzar el generador de escenarios con los estimadores. Estos comandos lanzan **experimentos de Monte Carlo** masivos que:
- Generan matrices dinámicas basadas en los escenarios descritos.
- Calculan la *Población Real Asintótica* de esa matriz.
- Evalúan si el Intervalo de Confianza generado (por Simulación Ponderada o Bootstrap Clásico) logra "atrapar" o cubrir el valor poblacional real.

---

## 🛠️ Instalación

Puedes instalar la librería en tu entorno local (o entorno virtual) ejecutando el siguiente comando desde la raíz del proyecto (donde se encuentra el archivo `pyproject.toml`):
```bashpip install -e .
```

---

## 💻 Uso por Terminal (CLI) - Ejemplos Rápidos

La instalación te otorga acceso a **21 comandos de terminal**.

**1. Análisis Empírico Individual con Auditoría (Marco N)**

Ejecutar el Coeficiente Natural Ordinal (NO), auditar anomalías a 2.5 sigmas para detectar jueces problemáticos y exportar las réplicas:

Bash

```
nmetrics-no encuestas_clinicas.csv -u 2.5 --exportar output_replicas.csv
```

**2. Tabla Comparativa Global**

Analizar datos categóricos guardados con comas (la librería traduce automáticamente letras como 'A', 'B' a isomorfismos numéricos):

Bash

```
nmetrics-nominal diagnosticos.csv -k 7 -s "," -r 5000
```

**3. Lanzar un Stress Test de Cobertura Matemática**

Simular 500 paneles de 10 jueces evaluando a 100 sujetos en escala de 5 para poner a prueba el Coeficiente Natural Intervalar (NI), guardando los resultados:

Bash

```
nmetrics-ni-cob -n 100 -m 10 -k 5 -e 500 -o validacion_ni.csv
```

---

## 🐍 Uso de la API en Python (Jupyter / Scripts)

Si prefieres integrar `nmetrics` en tus pipelines, la librería expone una arquitectura limpia de núcleos matemáticos y simuladores.

Python

```
import numpy as np
from nmetrics.interval import calcular_estadisticas_ni
from nmetrics.simulation.generador_escenarios import generar_matriz_dinamica

# 1. Simular un panel clínico de 30 pacientes, 5 médicos, escala 1-10 en escenario 'Razonable'
matriz_simulada = generar_matriz_dinamica(n_sujetos=30, m_jueces=5, k_escala=10, 
                                          escenario='Razonable', tipo_escala='paramétrica')

# 2. Ejecutar el motor de Simulación Ponderada Intervalar
# Retorna: Muestra, Ponderado, Límite Inf, Límite Sup, Réplicas, Matrices, Índices
ni_mu, ni_pob, ic_inf, ic_sup, *_ = calcular_estadisticas_ni(matriz_simulada, S_replicas=2000, k_escala=10, metodo_ic='SP')

print(f"Acuerdo Empírico (Muestra): {ni_mu:.4f}")
print(f"Acuerdo Poblacional (Proyectado): {ni_pob:.4f}")
print(f"Intervalo de Confianza del Panel: [{ic_inf:.4f}, {ic_sup:.4f}]")
```

---

### Notas para el Investigador:

- Todos los motores de inferencia toleran **datos faltantes (NaN)** de forma nativa.

- En la topología **Ordinal**, la auditoría de anomalías topológicas evalúa a las columnas (Jueces), mientras que en las familias **Intervalar** y **Nominal** evalúa a las filas (Sujetos).

- Los Coeficientes Naturales incluyen un Nivel III que calcula analíticamente la *Esperanza Nula* (Límite de Azar Termodinámico) exacto según la combinatoria del hiperespacio, informando de la validez topológica del consenso.

```

```
