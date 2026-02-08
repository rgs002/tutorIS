# src/rag_engine/generation/llm_client.py

import os
import google.generativeai as genai

#  Modelos de Google Gemini Disponibles (Referencia)
# Para configurar el modelo a usar, establece la variable de entorno GOOGLE_MODEL_NAME.
#
# Modelos descubiertos:
# - "gemini-2.5-flash"
# - "gemini-2.5-pro"
# - "gemini-2.0-flash"      (Default si no se especifica)
# - "gemini-2.0-flash-lite"
# ---------------------------------------------------------

class GeminiClient:
    """
    Un cliente para interactuar con el LLM de Google, en nuestro caso, Gemini.
    Lo he diseñado para que cargue la configuración una sola vez y sea fácil de reutilizar.
    """
    def __init__(self):
        """
        Al iniciar, configuramos la API y el modelo que vamos a usar.
        El modelo se lee desde la variable de entorno GOOGLE_MODEL_NAME.
        """
        # Primero, me aseguro de que la API key esté disponible.
        # El código espera encontrarla en una variable de entorno.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No se encontró la GOOGLE_API_KEY en las variables de entorno.")

        genai.configure(api_key=api_key)

        # Aquí defino la configuración para las peticiones al modelo.
        # He puesto una 'temperature' baja (0.1) para que las respuestas del LLM
        # sean más consistentes y predecibles
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            top_p=1,
            top_k=1,
            max_output_tokens=4096,
        )

        # Leo el modelo desde una variable de entorno
        # Si no está definida, uso un valor por defecto robusto.
        model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")

        # Inicializo el modelo con la configuración que acabamos de crear.
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,

        )
        print(f"Cliente Gemini inicializado con el modelo: {model_name}")

    def generate_text(self, prompt: str) -> str:
        """
        La función principal para enviar un prompt y obtener una respuesta.
        """
        try:
            # Le pasamos el prompt directamente al modelo.
            response = self.model.generate_content(prompt)
            # Devuelvo el texto de la respuesta, limpiando espacios por si acaso.
            return response.text.strip()
        except Exception as e:
            # Si algo va mal con la API
            print(f"Error al generar contenido con Gemini: {e}")
            return ""
