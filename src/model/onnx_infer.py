"""
Wrapper de inferência ONNX com a mesma interface do `Pipeline` sklearn
(`predict`, `predict_proba`, `classes_`), para a API poder trocar de backend
sem `if`s espalhados.

O modelo ONNX já embute o `TfidfVectorizer`, então a entrada é o texto já limpo
(mesmo `clean_text` do treino), exatamente como no `.pkl`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class OnnxClassifier:
    def __init__(self, model_path: str | Path, classes: Iterable[int] | None = None):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        # colunas de predict_proba seguem a ordem das classes ordenadas
        self.classes_ = np.asarray(
            sorted(classes) if classes is not None else [1, 2, 3, 4, 5]
        )

    def _run(self, texts: Iterable[str]) -> list:
        arr = np.array(list(texts), dtype=object).reshape(-1, 1)
        return self._session.run(None, {self._input_name: arr})

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self._run(texts)[0])

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self._run(texts)[1])
