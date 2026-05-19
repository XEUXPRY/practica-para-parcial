📊 📘 Diccionario de Datos

Dataset: Apartment Rentals + Crime & Socioeconomic Variables

🔹 1. Variables de vivienda (Apartment Rent)
🏠 Características del inmueble
variables = price
Tipo: Numérica continua
Unidad: USD (o moneda del dataset)
Descripción: Precio de alquiler del apartamento (variable dependiente Y)
variables = square_feet
Tipo: Numérica continua
Unidad: pies cuadrados
Descripción: Área del apartamento
variables = bedrooms
Tipo: Numérica discreta
Descripción: Número de habitaciones
variables = bathrooms
Tipo: Numérica continua/discreta
Descripción: Número de baños
🧾 Características cualitativas
variables = amenities
Tipo: Categórica binaria
Valores: basic, luxury
Descripción: Nivel de amenidades del apartamento
variables = pets_allowed
Tipo: Binaria
Valores: True / False
Descripción: Indica si se permiten mascotas
variables = has_photo
Tipo: Binaria
Valores: True / False
Descripción: Indica si el anuncio tiene foto
variables = source
Tipo: Categórica nominal
Descripción: Plataforma/origen del anuncio
🌍 Ubicación
variables = cityname
Tipo: Categórica nominal
Descripción: Ciudad donde se ubica el inmueble
variables = state
Tipo: Categórica nominal
Descripción: Estado o región
variables = latitude
Tipo: Numérica continua
Rango: [-90, 90]
Descripción: Coordenada geográfica (latitud)
variables = longitude
Tipo: Numérica continua
Rango: [-180, 180]
Descripción: Coordenada geográfica (longitud)
⏱️ Temporal
variables = time
Tipo: Numérica (timestamp Unix)
Descripción: Fecha de publicación del anuncio
Transformación recomendada: convertir a fecha (año, mes)
🔹 2. Variables socioeconómicas y crimen
👥 Demografía
variables = population
Tipo: Numérica continua
Descripción: Población promedio del área
racePctBlack, racePctWhite, racePctAsian, racePctHisp
Tipo: Numérica continua
Rango: [0,1] o porcentaje
Descripción: Proporción de población por grupo étnico
💰 Ingresos
variables = medIncome
Tipo: Numérica continua
Descripción: Ingreso medio de la zona
variables = medFamInc
Tipo: Numérica continua
Descripción: Ingreso medio familiar
🚓 Criminalidad (tasas por población)

(todas son numéricas continuas)

murdPerPop → homicidios
rapesPerPop → violaciones
robbbPerPop → robos
assaultPerPop → asaltos
burglPerPop → robos a vivienda
larcPerPop → hurtos
autoTheftPerPop → robo de vehículos
arsonsPerPop → incendios provocados
📊 Variables agregadas
ViolentCrimesPerPop
Tipo: Numérica continua
Descripción: Tasa total de crímenes violentos
nonViolPerPop
Tipo: Numérica continua
Descripción: Tasa de crímenes no violentos
avg_crime
Tipo: Numérica continua
Descripción: Promedio agregado de criminalidad