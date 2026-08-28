"""
Engenharia de atributos: vetorização TF-IDF dos laudos médicos.

Expõe apenas a configuração do vetorizador. A montagem final
(``TfidfVectorizer`` + classificador em um único ``Pipeline`` sklearn) é feita
em ``src/model/train.py`` para que o artefato salvo já contenha a vetorização
— o serviço de inferência só precisa de ``pipeline.predict([texto])``.
"""

from __future__ import annotations

import logging

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# min_df=5 remove termos raríssimos (ruído/erros de digitação);
# max_df=0.9 remove termos onipresentes; bigramas capturam expressões clínicas
# ("blood pressure", "nervous system"); sublinear_tf amortece contagens altas.
DEFAULT_TFIDF_PARAMS: dict = dict(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    sublinear_tf=True,
)


def build_vectorizer(**overrides) -> TfidfVectorizer:
    """Cria um ``TfidfVectorizer`` com os parâmetros padrão do projeto."""
    params = {**DEFAULT_TFIDF_PARAMS, **overrides}
    logger.info("TfidfVectorizer params: %s", params)
    return TfidfVectorizer(**params)
