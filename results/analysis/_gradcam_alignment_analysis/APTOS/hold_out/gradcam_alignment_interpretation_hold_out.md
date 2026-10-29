# Informe de alineación Grad-CAM — hold_out

Este informe clasifica patrones de atención para priorizar revisión. Una predicción correcta puede aparecer como sospechosa si la energía Grad-CAM cae fuera de retina o sobre el borde. Eso es compatible con aprendizaje por atajos, pero no lo demuestra por sí solo.

## Umbrales utilizados

- Atención extrarretinal elevada: outside_energy ≥ 0.500.
- Atención extrarretinal dominante: outside_energy ≥ 0.700.
- Borde elevado: border_ratio ≥ 0.436.
- Periferia elevada: peripheral_attention ≥ 0.400.

Los umbrales automáticos se calculan de forma conjunta sobre todas las variantes del split, para conservar comparabilidad.

## Resumen por variante

- **adaptive_feather**: outside=0.427, border=0.360, central=0.401, shortcut sospechado=39.9%, aciertos sospechosos entre aciertos=40.6%.
- **fixed_feather**: outside=0.430, border=0.358, central=0.402, shortcut sospechado=40.9%, aciertos sospechosos entre aciertos=41.1%.
- **hard_mask**: outside=0.358, border=0.307, central=0.472, shortcut sospechado=30.3%, aciertos sospechosos entre aciertos=30.9%.
- **no_mask**: outside=0.347, border=0.299, central=0.498, shortcut sospechado=33.3%, aciertos sospechosos entre aciertos=34.3%.

## Lectura comparativa

- Menor energía extrarretinal promedio: **no_mask**.
- Menor dependencia promedio del borde: **no_mask**.
- Mayor energía retinal central: **no_mask**.
- En clase 0 correcta, 30.4% presenta patrón sospechoso. Parte puede relacionarse con evidencia global de ausencia de lesión; sin embargo, una activación repetida sobre fondo o borde sigue siendo metodológicamente relevante.

## Indicación metodológica

1. No eliminar imágenes automáticamente por Grad-CAM. Usar las banderas para priorizar revisión.
2. Comparar el mismo filename entre las cuatro variantes. Un cambio sistemático de fondo a borde o de borde a retina es más informativo que un mapa aislado.
3. Separar análisis por clase, luminosidad y acierto/error. Clase 0 requiere especial cautela porque no hay una lesión positiva obligatoria.
4. Reportar proporción de predicciones correctas con shortcut sospechado; accuracy o QWK por sí solas no garantizan razonamiento anatómicamente plausible.
5. Confirmar una muestra de casos con mapas crudos, otra técnica de atribución y revisión clínica antes de afirmar causalidad.

## Limitaciones

Grad-CAM es post-hoc, depende de la capa elegida y tiene resolución espacial limitada. La máscara usada para cuantificar dentro/fuera es aproximada, especialmente en no_mask. Por ello, estos indicadores expresan plausibilidad y riesgo de atajo, no una explicación causal definitiva.
