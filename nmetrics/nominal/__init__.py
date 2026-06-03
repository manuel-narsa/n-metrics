"""
Módulo Nominal de n-metrics.
Contiene los núcleos termodinámicos (Marco N) y estimadores clásicos
para variables categóricas puras sin orden (escala nominal).
"""

# 1. Importaciones del motor Termodinámico (Marco N)
from .nn_core import (
    detectar_anomalias_nn,
    calcular_azar_termodinamico_nn,
    calcular_estadisticas_nn_unificada,
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

__all__ = [
    "detectar_anomalias_nn",
    "calcular_azar_termodinamico_nn",
    "calcular_estadisticas_nn_unificada",
    "calcular_percentil_universal_nn",
    "calcular_estadisticas_akn",
    "calcular_akn_poblacion_asintotica",
    "calcular_estadisticas_kf",
    "calcular_kf_poblacion_asintotica"
]