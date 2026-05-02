"""
Módulo Ordinal de nmetrics.
Expone las funciones principales para calcular NO, Alpha de Krippendorff Ordinal (AKO) 
y W de Kendall (W).
"""

# 1. Importaciones del motor Natural Ordinal (NO)
from .no_core import (
    calcular_estadisticas_no,
    detectar_anomalias_no,
    calcular_azar_termodinamico_no_analitico_exacto,
    calcular_percentil_universal_no_exacto
)

# 2. Importaciones del motor Alpha de Krippendorff Ordinal (AKO)
from .ako_core import (
    calcular_estadisticas_ako,
    calcular_ako_poblacion_asintotica
)

# 3. Importaciones del motor W de Kendall (W)
from .w_core import (
    calcular_estadisticas_w,
    calcular_w_poblacion_asintotica
)

# Definimos explícitamente qué se exporta si alguien usa "from nmetrics.ordinal import *"
__all__ = [
    "calcular_estadisticas_no",
    "detectar_anomalias_no",
    "calcular_azar_termodinamico_no_analitico_exacto",
    "calcular_percentil_universal_no_exacto",
    "calcular_estadisticas_ako",
    "calcular_ako_poblacion_asintotica",
    "calcular_estadisticas_w",
    "calcular_w_poblacion_asintotica"
]