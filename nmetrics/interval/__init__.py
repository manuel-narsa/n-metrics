"""
Módulo Intervalar de nmetrics.
Expone las funciones principales para calcular NI, AKI e ICC(2,1).
"""

# 1. Importaciones del motor Natural Intervalar (NI)
from .ni_core import (
    calcular_estadisticas_ni,
    detectar_anomalias_ni,
    calcular_azar_termodinamico_ni,
    calcular_percentil_universal_ni
)

# 2. Importaciones del motor Alpha de Krippendorff Intervalar (AKI)
from .aki_core import (
    calcular_estadisticas_aki,
    calcular_aki_poblacion_asintotica
)

# 3. Importaciones del motor ICC(2,1)
from .icc21_core import (
    calcular_estadisticas_icc21,
    calcular_icc_poblacion_asintotica
)

# Definimos explícitamente qué se exporta si alguien usa "from nmetrics.interval import *"
__all__ = [
    "calcular_estadisticas_ni",
    "detectar_anomalias_ni",
    "calcular_azar_termodinamico_ni",
    "calcular_percentil_universal_ni",
    "calcular_estadisticas_aki",
    "calcular_aki_poblacion_asintotica",
    "calcular_estadisticas_icc21",
    "calcular_icc_poblacion_asintotica"
]