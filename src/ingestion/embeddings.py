import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingFactory:
    """
    Configurar instancias de modelos de embeddings.

    Esta clase centraliza la lógica para seleccionar e instanciar el cliente
    de embeddings apropiado según la configuración del entorno.
    """

    @staticmethod
    def get_embeddings() -> Embeddings:
        """
        Carga la configuración desde variables de entorno y devuelve la instancia
        del modelo de embeddings de LangChain correspondiente.

        Utiliza `dotenv` para cargar las variables.

        Variables de entorno utilizadas:
        - `EMBEDDING_PROVIDER`: El proveedor a utilizar ("huggingface"). Por defecto, "huggingface".
        - `EMBEDDING_MODEL_HF`: El nombre del modelo de HuggingFace. Por defecto,
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".

        Raises:
            ValueError: Si el proveedor de embeddings no es reconocido.

        Returns:
            Una instancia compatible con la interfaz `Embeddings` de LangChain.
        """
        load_dotenv()
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")

        if provider == "huggingface":
            model_name = os.getenv("EMBEDDING_MODEL_HF", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        else:
            raise ValueError(f"Proveedor de embeddings no soportado: {provider}")