"""
Treinamento e predição com KNN para reconhecimento facial.

Esta parte é baseada no exemplo clássico de KNN da biblioteca
`face_recognition` (https://github.com/ageitgey/face_recognition).

Descrição do algoritmo
----------------------
O classificador KNN é primeiro treinado em um conjunto de faces rotuladas
(conhecidas), e em seguida consegue predizer a pessoa em uma imagem
desconhecida encontrando as k faces mais similares (imagens com os
embeddings faciais mais próximos sob a distância euclidiana) no seu
conjunto de treino, e fazendo uma votação por maioria (possivelmente
ponderada) sobre os rótulos dessas faces.

Por exemplo, se k=3 e as três imagens de face mais próximas da imagem
dada no conjunto de treino forem uma imagem de Biden e duas imagens de
Obama, o resultado seria "Obama".

Esta implementação usa votação ponderada (`weights='distance'`), de modo
que os votos dos vizinhos mais próximos têm peso maior.

Fluxo do módulo
---------------
1. `train()` localiza a face em cada imagem do conjunto de treino,
   extrai o embedding facial e treina um classificador KNN com os pares
   (embedding, rótulo da pessoa).
2. `predict()` recebe uma imagem nova, localiza a face, extrai o embedding
   e pergunta ao KNN qual é o vizinho mais próximo, aceitando a predição
   somente se a distância estiver abaixo do limiar configurado.
3. `show_prediction_labels_on_image()` desenha as caixas e os rótulos
   sobre a imagem original e salva o resultado.

Convenção das pastas
--------------------
- `dir_Treino`: alimenta o treino do KNN (lida por `train()`);
- `dir_Testes`: imagens cuja identidade o modelo precisa predizer
  (lidas por `predict()`).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw
from sklearn import neighbors

from dataset_utils import ALLOWED_EXTENSIONS

MODEL_FILENAME = "trained_knn_model.clf"


def train(
    train_dir: Path,
    model_save_path: Path | None = None,
    n_neighbors: int | None = None,
    knn_algo: str = "ball_tree",
    verbose: bool = False,
):
    """Treina um classificador KNN a partir de uma pasta com subpastas por pessoa.

    Estrutura esperada:

        train_dir/
          pessoa_1/
            imagem1.jpg
          pessoa_2/
            imagem2.jpg
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Pasta de treino não encontrada: {train_dir}")

    X = []
    y = []

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path_as_text in image_files_in_folder(str(class_dir)):
            image_path = Path(image_path_as_text)
            image = face_recognition.load_image_file(image_path)              # reconhece
            face_bounding_boxes = face_recognition.face_locations(image)      # reconhece

            # Para o treino, exigimos exatamente uma face por imagem.
            # Imagens sem face ou com mais de uma face são ignoradas.
            if len(face_bounding_boxes) != 1:
                if verbose:
                    motivo = ("nenhuma face encontrada" if len(face_bounding_boxes) == 0
                              else "mais de uma face encontrada")
                    print(f"Imagem ignorada no treino: {image_path} ({motivo})")
                continue

            # Adiciona o encoding da face ao conjunto de treino
            encoding = face_recognition.face_encodings(
                image, known_face_locations=face_bounding_boxes
            )[0]
            X.append(encoding)
            y.append(label)

    if not X:
        raise RuntimeError(
            "Nenhuma face válida foi encontrada para treinamento. "
            "Verifique se cada imagem possui exatamente uma face."
        )

    # Define o número de vizinhos automaticamente, se não informado
    if n_neighbors is None:
        n_neighbors = max(1, round(math.sqrt(len(X))))
        if verbose:
            print(f"Chose n_neighbors automatically: {n_neighbors}")

    # Evita erro quando o usuário define k maior que a quantidade de imagens válidas
    n_neighbors = min(n_neighbors, len(X))

    # Cria e treina o classificador KNN
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm=knn_algo,
        weights="distance",
    )
    knn_clf.fit(X, y)

    # Salva o classificador treinado
    if model_save_path is not None:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        with model_save_path.open("wb") as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(
    X_img_path: Path,
    knn_clf=None,
    model_path: Path | None = None,
    distance_threshold: float = 0.6,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Reconhece faces em uma imagem usando um classificador KNN treinado.

    Para faces de pessoas não reconhecidas, o nome `unknown` é retornado.
    A localização segue o padrão da biblioteca face_recognition:
    `(top, right, bottom, left)`.
    """
    if not X_img_path.is_file() or X_img_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid image path: {X_img_path}")

    if knn_clf is None and model_path is None:
        raise ValueError("Must supply knn classifier either through knn_clf or model_path")

    # Carrega o modelo treinado (se informado por caminho)
    if knn_clf is None:
        with model_path.open("rb") as f:
            knn_clf = pickle.load(f)

    # Carrega a imagem e localiza as faces
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # Se nenhuma face foi encontrada, retorna lista vazia
    if not X_face_locations:
        return []

    # Calcula os encodings das faces da imagem de teste
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations
    )

    # Usa o KNN para encontrar a melhor correspondência
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold
        for i in range(len(X_face_locations))
    ]

    # Prediz as classes e descarta as que ficam fora do limiar
    return [
        (pred, loc) if rec else ("unknown", loc)
        for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)
    ]


def show_prediction_labels_on_image(
    img_path: Path,
    predictions: Iterable[tuple[str, tuple[int, int, int, int]]],
    output_path: Path,
) -> None:
    """Salva a imagem com caixas e rótulos desenhados sobre as faces."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as source_image:
        pil_image = source_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Desenha a caixa ao redor da face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)

        # `textbbox` é compatível com versões novas do Pillow
        # (substitui `textsize`, que foi removida)
        text = str(name)
        bbox = draw.textbbox((left, bottom), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Desenha o rótulo abaixo da face
        draw.rectangle(
            ((left, bottom - text_height - 10), (left + text_width + 12, bottom)),
            fill=(0, 0, 255),
            outline=(0, 0, 255),
        )
        draw.text((left + 6, bottom - text_height - 5), text, fill=(255, 255, 255))

    pil_image.save(output_path)
