"""
Módulo Nominal de nmetrics.
Expone las funciones principales para calcular NN, Alpha de Krippendorff Nominal (AKN) 
y Kappa de Fleiss (KF).
"""

# 1. Importaciones del motor Natural Nominal (NN)
from .nn_core import (
    calcular_estadisticas_nn,
    detectar_anomalias_nn,
    calcular_azar_termodinamico_nn,
    calcular_percentil_universal_nn
)

# 2. Importaciones del motor Alpha de Krippendorff Nominal (AKN)
from .akn_core import (
    calcular_estadisticas_akn,
    calcular_akn_poblacion_asintotica
)

# 3. Importaciones del motor Kappa de Fleiss (KF)
from .kf_core import (
    calcular_estadisticas_kf,
    calcular_kf_poblacion_asintotica
)

# Definimos explícitamente qué se exporta si alguien usa "from nmetrics.nominal import *"
__all__ = [
    "calcular_estadisticas_nn",
    "detectar_anomalias_nn",
    "calcular_azar_termodinamico_nn",
    "calcular_percentil_universal_nn",
    "calcular_estadisticas_akn",
    "calcular_akn_poblacion_asintotica",
    "calcular_estadisticas_kf",
    "calcular_kf_poblacion_asintotica"
]