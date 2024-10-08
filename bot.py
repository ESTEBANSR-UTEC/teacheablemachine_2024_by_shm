#!/usr/bin/env python
# pylint: disable=unused-argument
from PIL import Image, ImageOps
import logging
import os
import google.generativeai as genai
import numpy as np
from keras.models import load_model

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Configurar la API de Gemini
API_KEY = 'AIzaSyCw863r8j_-MBbtSHpNgLcBRrtxvdJo_38'  # Reemplaza con tu propia clave de API
os.environ['API_KEY'] = API_KEY
genai.configure(api_key=os.environ['API_KEY'])

# Cargar el modelo de Keras
model = load_model("models/keras_model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()

# Logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Funciones del bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enviar un mensaje cuando se emite el comando /start."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hola {user.mention_html()}! Soy un bot que usa Google Generative AI y también puedo predecir imágenes.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enviar un mensaje cuando se emite el comando /help."""
    await update.message.reply_text("Envía una imagen o mensaje y te responderé con predicciones o usando Gemini AI!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Recibe un mensaje de texto y genera una respuesta usando Google Generative AI."""
    texto_recibido = update.message.text
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(texto_recibido)
    texto_generado = response.text
    await update.message.reply_text(texto_generado)

async def images_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Procesa las imágenes enviadas por los usuarios."""
    try:
        # Obtén el archivo de imagen enviado por el usuario
        image_file = await context.bot.get_file(update.message.photo[-1].file_id)
        file_path = "image.jpg"
        await image_file.download_to_drive(file_path)

        # Abre la imagen con Pillow
        with Image.open(file_path) as img:
            # Convertir la imagen a blanco y negro
            img_bw = img.convert("L")
            bw_image_path = "image_bw.jpg"
            img_bw.save(bw_image_path)

            # Redimensionar la imagen para la predicción con Keras
            size = (224, 224)
            img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(img_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predicción de la clase
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

        # Enviar la imagen en blanco y negro al usuario
        with open(bw_image_path, 'rb') as bw_img:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=bw_img)

        # Enviar la predicción de la imagen
        await update.message.reply_text(f"Clase predicha: {class_name.strip()} con una confianza de {confidence_score:.2f}")

    except Exception as error:
        print(f"Ups, algo falló: {error}")
        await update.message.reply_text("Hubo un error al procesar la imagen.")

# Iniciar el bot
def main() -> None:
    """Iniciar el bot."""
    application = Application.builder().token("7085811612:AAER371wIxTMJDpW3O70J6XW8A4m6L1aTcs").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, images_handler))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

