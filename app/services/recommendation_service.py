"""Deterministic monitoring drivers and non-prescriptive recommendations."""

from __future__ import annotations

from typing import Any


DISCLAIMER = (
    "Информационная рекомендация для мониторинга; не заменяет агрономическое "
    "заключение и полевые измерения."
)


def _percent(value: float) -> str:
    return f"{abs(value):.0%}"


def generate_recommendations(observation: dict[str, Any]) -> dict[str, Any]:
    """Build stable decision-support text from already calculated values only."""
    drivers: list[dict[str, str]] = []
    recommendations: list[dict[str, str]] = []
    precipitation_anomaly = observation.get("precipitation_anomaly")
    temperature_anomaly = observation.get("temperature_anomaly")
    heat_stress = observation.get("heat_stress")
    ndvi_anomaly = observation.get("ndvi_anomaly")
    vegetation_stress = observation.get("vegetation_stress")
    drought_fraction = observation.get("drought_fraction")
    quality = observation.get("data_quality")

    if quality == "segmentation_only":
        drivers.append({"key": "data_quality", "severity": "info", "title": "Ограниченное качество данных", "detail": "Доступны только результаты спутниковой сегментации; погодные и растительные индикаторы недоступны."})
    elif quality == "partial":
        missing = []
        if observation.get("soil_moisture_stress") is None:
            missing.append("влажность почвы")
        if observation.get("vegetation_stress") is None:
            missing.append("индикатор растительности")
        if observation.get("rainfall_stress") is None:
            missing.append("стресс осадков")
        missing_text = ", ".join(missing) if missing else "часть дополнительных индикаторов"
        drivers.append({"key": "data_quality", "severity": "warning", "title": "Частичное качество данных", "detail": f"Для оценки пока недоступны: {missing_text}."})
    else:
        drivers.append({"key": "data_quality", "severity": "info", "title": "Поддержанные индикаторы доступны", "detail": "Все текущие поддержанные компоненты оценки доступны; полевые измерения всё равно не заменяются."})

    if precipitation_anomaly is not None and precipitation_anomaly <= -0.20:
        drivers.append({"key": "rainfall_deficit", "severity": "danger" if precipitation_anomaly <= -0.40 else "warning", "title": "Дефицит осадков", "detail": f"Осадки за месяц были примерно на {_percent(precipitation_anomaly)} ниже локальной исторической базы."})
    if precipitation_anomaly is not None and precipitation_anomaly <= -0.40:
        recommendations.extend([
            {"key": "check_root_zone_moisture", "priority": "high", "title": "Проверить влажность почвы в корневой зоне", "detail": "Сопоставьте спутниковую оценку с фактическим измерением влажности почвы на участке."},
            {"key": "repeat_analysis_7_days", "priority": "high", "title": "Повторить анализ через 7 дней", "detail": "Сравните новое наблюдение с текущим, чтобы отследить изменение показателей."},
        ])

    if (heat_stress is not None and heat_stress >= 0.35) or (temperature_anomaly is not None and temperature_anomaly >= 2.0):
        drivers.append({"key": "heat_stress", "severity": "warning", "title": "Тепловой стресс", "detail": "Температурный индикатор указывает на повышенную нагрузку относительно доступной базы."})
        recommendations.append({"key": "inspect_heat_stress", "priority": "medium", "title": "Проверить признаки теплового и водного стресса", "detail": "Сопоставьте наблюдение с состоянием растений и измерениями на участке."})

    if (vegetation_stress is not None and vegetation_stress >= 0.35) or (ndvi_anomaly is not None and ndvi_anomaly <= -0.07):
        drivers.append({"key": "vegetation_decline", "severity": "warning", "title": "Снижение растительного индикатора", "detail": "NDVI или рассчитанный стресс растительности указывает на снижение относительно локальной исторической базы."})
        recommendations.append({"key": "inspect_vegetation", "priority": "high", "title": "Провести полевую проверку растений", "detail": "Проверьте состояние растительности в нескольких репрезентативных точках участка."})

    if drought_fraction is not None and drought_fraction >= 0.35:
        drivers.append({"key": "satellite_drought_pattern", "severity": "warning", "title": "Спутниковый паттерн засухи", "detail": f"Спутниковая сегментация обнаружила признаки засухи примерно на {drought_fraction:.1%} участка."})

    if observation.get("risk_level") in {"medium", "high"} or (precipitation_anomaly is not None and precipitation_anomaly <= -0.40) or (drought_fraction is not None and drought_fraction >= 0.35):
        recommendations.append({"key": "field_check", "priority": "high", "title": "Провести полевую проверку в течение 3–7 дней", "detail": "Сопоставьте текущую оценку с состоянием растений и фактической влажностью почвы."})
    if observation.get("risk_level") == "high":
        recommendations.append({"key": "irrigation_planning", "priority": "medium", "title": "Подготовить план проверки орошения", "detail": "При наличии системы орошения подготовьте план полива после проверки фактической влажности почвы."})
    if observation.get("trend") == "worsening":
        recommendations.append({"key": "monitor_worsening", "priority": "medium", "title": "Отслеживать динамику", "detail": "Повторите спутниковый анализ через 7–14 дней и сравните показатели."})
    if not recommendations:
        recommendations.append({"key": "routine_monitoring", "priority": "low", "title": "Продолжить регулярный мониторинг", "detail": "Сохраняйте наблюдения и сопоставляйте следующие измерения с текущим результатом."})

    return {"drivers": drivers, "recommendations": recommendations, "disclaimer": DISCLAIMER}