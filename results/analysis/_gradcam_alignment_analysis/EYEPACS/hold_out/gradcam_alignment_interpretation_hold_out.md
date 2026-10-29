# Informe de alineación Grad-CAM — hold_out

Este informe clasifica patrones de atención para priorizar revisión. Una predicción correcta puede aparecer como sospechosa si la energía Grad-CAM cae fuera de retina o sobre el borde. Eso es compatible con aprendizaje por atajos, pero no lo demuestra por sí solo.

## Umbrales utilizados

- Atención extrarretinal elevada: outside_energy ≥ 0.500.
- Atención extrarretinal dominante: outside_energy ≥ 0.700.
- Borde elevado: border_ratio ≥ 0.434.
- Periferia elevada: peripheral_attention ≥ 0.400.

Los umbrales automáticos se calculan de forma conjunta sobre todas las variantes del split, para conservar comparabilidad.

## Resumen por variante

- **adaptive_feather**: outside=0.390, border=0.331, central=0.438, shortcut sospechado=35.8%, aciertos sospechosos entre aciertos=39.0%.
- **fixed_feather**: outside=0.367, border=0.326, central=0.455, shortcut sospechado=31.8%, aciertos sospechosos entre aciertos=33.4%.
- **hard_mask**: outside=0.404, border=0.358, central=0.426, shortcut sospechado=39.7%, aciertos sospechosos entre aciertos=46.0%.
- **no_mask**: outside=0.370, border=0.339, central=0.455, shortcut sospechado=34.5%, aciertos sospechosos entre aciertos=39.5%.

## Lectura comparativa

- Menor energía extrarretinal promedio: **fixed_feather**.
- Menor dependencia promedio del borde: **fixed_feather**.
- Mayor energía retinal central: **no_mask**.
- **fixed_feather** reduce simultáneamente energía fuera de retina y concentración de borde respecto a hard_mask, lo que apoya la hipótesis de que una transición suave disminuye pistas espurias periféricas.
- **adaptive_feather** reduce simultáneamente energía fuera de retina y concentración de borde respecto a hard_mask, lo que apoya la hipótesis de que una transición suave disminuye pistas espurias periféricas.
- El feather adaptativo no mejora estas métricas frente al fijo en este split. Conviene revisar umbrales de feather, categorías de luminosidad y casos extremos.
- En clase 0 correcta, 44.4% presenta patrón sospechoso. Parte puede relacionarse con evidencia global de ausencia de lesión; sin embargo, una activación repetida sobre fondo o borde sigue siendo metodológicamente relevante.

## Indicación metodológica

1. No eliminar imágenes automáticamente por Grad-CAM. Usar las banderas para priorizar revisión.
2. Comparar el mismo filename entre las cuatro variantes. Un cambio sistemático de fondo a borde o de borde a retina es más informativo que un mapa aislado.
3. Separar análisis por clase, luminosidad y acierto/error. Clase 0 requiere especial cautela porque no hay una lesión positiva obligatoria.
4. Reportar proporción de predicciones correctas con shortcut sospechado; accuracy o QWK por sí solas no garantizan razonamiento anatómicamente plausible.
5. Confirmar una muestra de casos con mapas crudos, otra técnica de atribución y revisión clínica antes de afirmar causalidad.

## Limitaciones

Grad-CAM es post-hoc, depende de la capa elegida y tiene resolución espacial limitada. La máscara usada para cuantificar dentro/fuera es aproximada, especialmente en no_mask. Por ello, estos indicadores expresan plausibilidad y riesgo de atajo, no una explicación causal definitiva.
