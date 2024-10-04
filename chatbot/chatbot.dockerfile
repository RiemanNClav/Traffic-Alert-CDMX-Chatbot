# Usa una imagen base de Rasa
FROM rasa/rasa:3.6.20



# Copia los archivos de tu proyecto al contenedor
COPY . /app

# Define el directorio de trabajo
WORKDIR /app

USER root


# Instala las dependencias adicionales si es necesario (puedes agregar más librerías en el requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt


# Exponer los puertos necesarios (5005 para Rasa y 5055 para las acciones)
EXPOSE 5005

# Comando para entrenar el modelo (opcional, si quieres entrenar cada vez que creas el contenedor)
# RUN rasa train

# Comando para ejecutar el servidor de Rasa
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"]