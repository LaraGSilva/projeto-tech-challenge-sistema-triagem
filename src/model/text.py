"""
Normalização de texto compartilhada entre treino e inferência.

Módulo folha, sem dependências pesadas (só ``re``), para poder ser importado
tanto pela pipeline de treino quanto pela API de inferência sem arrastar
pandas / sklearn.
"""

from __future__ import annotations

import re

_BRACKETS = re.compile(r"\[.*?\]")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTISPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normaliza um laudo: minúsculas, sem conteúdo entre colchetes, sem dígitos
    e sem pontuação. Retorna string única com espaços simples.

    Precisa ser idêntico ao usado no treino — o modelo foi ajustado sobre o
    texto já limpo.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _BRACKETS.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _MULTISPACE.sub(" ", text).strip()
