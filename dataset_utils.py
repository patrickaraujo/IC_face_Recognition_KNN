"""
Funções auxiliares para preparar pastas, copiar imagens, dividir os
conjuntos de treino e teste, rotacionar imagens entre execuções e salvar logs.

Semântica das pastas (convenção usual de ML):

- `dir_Treino`: pasta que alimenta o **treino** do classificador KNN.
- `dir_Testes`: pasta com as imagens usadas para **teste/predição**.
- `porc`: percentual de imagens de cada classe que **permanece** em `dir_Treino`.
  O restante é movido para `dir_Testes`.
"""

from __future__ import annotations

import csv
import random
import shutil
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from models import Imagem, Pastas

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_image_file(path: Path) -> bool:
    """Retorna True quando o arquivo tem uma extensão de imagem aceita."""
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def ensure_empty_dir(path: Path, erase: bool = False) -> Path:
    """Garante que uma pasta exista.

    Quando `erase` é True, apaga a pasta anterior e cria uma nova vazia.
    Quando False, mantém o conteúdo existente.
    """
    if path.exists() and erase:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def split_integer(total: int, parts: int) -> list[int]:
    """Divide um inteiro em partes quase iguais.

    Exemplo: `split_integer(10, 3)` retorna `[3, 3, 4]`.
    Usada para decidir quantas imagens serão trocadas entre treino e teste
    em cada execução. Corresponde à função `divInt` do código original.
    """
    if total < 0:
        raise ValueError("total precisa ser maior ou igual a zero")
    if parts <= 0:
        raise ValueError("parts precisa ser maior que zero")

    base = total // parts
    remainder = total % parts
    result = [base] * parts

    # Distribui o resto nas últimas posições para preservar o comportamento
    # do código original.
    for index in range(parts - remainder, parts):
        if 0 <= index < parts:
            result[index] += 1

    return result


def resize_image(path: Path, width: int, height: int) -> None:
    """Redimensiona uma imagem e sobrescreve o arquivo original."""
    with Image.open(path) as image:
        resized = image.resize((width, height))
        resized.save(path, optimize=True)


def resize_by_divisor(path: Path, divisor: int, restore_original_size: bool) -> tuple[int, int]:
    """Reduz uma imagem usando um divisor e, opcionalmente, restaura o tamanho.

    O código original fazia duas chamadas: primeiro reduzia a imagem e depois,
    quando `reset=True`, voltava ao tamanho original. Esse comportamento foi
    preservado, embora a recompressão possa reduzir a qualidade da imagem.
    """
    if divisor <= 1:
        with Image.open(path) as image:
            return image.size

    with Image.open(path) as image:
        original_width, original_height = image.size

    reduced_width = max(1, original_width // divisor)
    reduced_height = max(1, original_height // divisor)
    resize_image(path, reduced_width, reduced_height)

    if restore_original_size:
        resize_image(path, original_width, original_height)

    return original_width, original_height


def renomeia_dataset(source_dir: Path, renamed_dir: Path, erase: bool = False) -> Path:
    """Copia o dataset e prefixa cada imagem com o nome da classe.

    Equivale à função `novoDirRenomeia` do código original.

    Entrada esperada:

        dataset/
          pessoa_1/
            img1.jpg
          pessoa_2/
            img1.jpg

    Saída:

        renamed_dataset/
          pessoa_1/
            pessoa_1_img1.jpg
          pessoa_2/
            pessoa_2_img1.jpg

    Isso evita conflito quando duas pessoas possuem arquivos com o mesmo nome.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {source_dir}")

    ensure_empty_dir(renamed_dir, erase=erase)

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        destination_class_dir = ensure_empty_dir(renamed_dir / label)

        for image_path in sorted(class_dir.iterdir()):
            if not is_image_file(image_path):
                continue

            new_name = f"{label}_{image_path.name}"
            shutil.copy2(image_path, destination_class_dir / new_name)

    return renamed_dir


def cria_dir_treino(
    source_dir: Path,
    dir_treino: Path,
    erase: bool = False,
    resize_divisor: int | None = None,
    restore_original_size: bool = True,
) -> list[Imagem]:
    """Cria `dir_Treino`, com todas as imagens do dataset organizadas por classe.

    Esta é a pasta que alimentará o treino do classificador KNN. O formato
    mantém uma subpasta por pessoa, que é o exigido pelo treino do KNN.

    Em seguida, `cria_dir_testes` move parte dessas imagens para `dir_Testes`,
    deixando `dir_Treino` apenas com as imagens efetivamente usadas no treino.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Pasta de amostras não encontrada: {source_dir}")

    ensure_empty_dir(dir_treino, erase=erase)
    aT: list[Imagem] = []

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        destination_class_dir = ensure_empty_dir(dir_treino / label)

        for image_path in sorted(class_dir.iterdir()):
            if not is_image_file(image_path):
                continue

            destination_path = destination_class_dir / image_path.name
            shutil.copy2(image_path, destination_path)

            if resize_divisor:
                resize_by_divisor(destination_path, resize_divisor, restore_original_size)

            aT.append(
                Imagem(
                    nomeO=image_path.name,
                    diretorioO=image_path,
                    pastaO=label,
                    nomeM=destination_path.name,
                    diretorioM=destination_path,
                    pastaM=label,
                    em_teste=False,
                    ja_testada=False,
                )
            )

    return aT


def _find_record_by_path(records: Sequence[Imagem], path: Path) -> Imagem:
    """Localiza o registro associado a uma imagem pelo caminho atual."""
    normalized_path = path.resolve()
    for record in records:
        if record.diretorioM.resolve() == normalized_path:
            return record
    raise LookupError(f"Registro não encontrado para a imagem: {path}")


def _split_counts(total_images: int, porc: int) -> tuple[int, int]:
    """Calcula quantas imagens permanecem em `dir_Treino` e quantas vão para `dir_Testes`.

    `porc` representa o percentual de imagens que **permanece** em treino.
    O restante é movido para teste.

    Para classes com mais de uma imagem, a função garante pelo menos uma
    imagem em cada pool, evitando que uma pessoa fique sem amostras em treino
    ou sem amostras em teste.
    """
    if total_images < 0:
        raise ValueError("total_images precisa ser maior ou igual a zero")
    if not 1 <= porc <= 99:
        raise ValueError("porc deve estar entre 1 e 99")

    if total_images == 0:
        return 0, 0
    if total_images == 1:
        # Com apenas uma imagem, mantém em treino.
        return 1, 0

    em_treino = ceil(total_images * porc / 100)
    em_treino = max(1, min(em_treino, total_images - 1))
    em_testes = total_images - em_treino
    return em_treino, em_testes


def cria_dir_testes(
    dir_treino: Path,
    porc: int,
    dir_testes: Path,
    nomeT: str,
    aT: list[Imagem],
    rng: random.Random,
    erase: bool = True,
) -> list[Imagem]:
    """Cria `dir_Testes` movendo `(100 - porc)%` das imagens de cada classe
    para fora de `dir_Treino`.

    - `dir_Treino` permanece com o conjunto que treinará o KNN;
    - `dir_Testes` recebe o subconjunto que será predito;
    - `porc` é o percentual de imagens que permanece em treino.
    """
    ensure_empty_dir(dir_testes, erase=erase)

    print("\n\nCriando diretório de testes...")
    print("Movendo as imagens...")

    i = 1
    tam = max(1, len(str(len(aT))))

    for class_dir in sorted(dir_treino.iterdir()):
        if not class_dir.is_dir():
            continue

        image_paths = [p for p in sorted(class_dir.iterdir()) if is_image_file(p)]
        _, para_testes = _split_counts(len(image_paths), porc)
        selecionadas = rng.sample(image_paths, para_testes) if para_testes else []

        for src_file in selecionadas:
            record = _find_record_by_path(aT, src_file)

            formato = src_file.suffix.lower().lstrip(".")
            novo_nome = f"{nomeT}_{str(i).zfill(tam)}.{formato}"
            dst_file = dir_testes / novo_nome

            shutil.move(str(src_file), dst_file)

            record.nomeM = novo_nome
            record.diretorioM = dst_file
            record.pastaM = dir_testes.name
            record.em_teste = True
            record.ja_testada = True

            i += 1

    return sorted(aT, key=lambda h: h.nomeO)


def pasta_info(aT: Sequence[Imagem]) -> list[Pastas]:
    """Calcula a quantidade de imagens por classe no estado atual.

    Equivale à função `pastaInfo` do código original.
    """
    folders = [record.pastaO for record in aT]
    em_dir_testes = [record.pastaO for record in aT if record.em_teste]
    pastas_unicas = sorted(set(folders))

    return [
        Pastas(nome=nome, qtd=folders.count(nome), qtd_testes=em_dir_testes.count(nome))
        for nome in pastas_unicas
    ]


def permuta(
    aT: list[Imagem],
    pastasL: Sequence[Pastas],
    inter: int,
    interA: int,
    rng: random.Random,
    dir_treino: Path,
    dir_testes: Path,
    amostras: Path,
    resize_divisor: int | None = None,
    restore_original_size: bool = True,
) -> None:
    """Troca imagens entre `dir_Testes` e `dir_Treino` para a próxima execução.

    Equivale à função `permuta` do código original. A ideia é parecida com
    uma validação cruzada simples: imagens que estavam em `dir_Testes`
    (já preditas) voltam para `dir_Treino`, e imagens que ainda não foram
    testadas saem de `dir_Treino` para `dir_Testes`.
    """
    for pasta in pastasL:
        arr_quantidades, not_ok = divInt(pasta.qtd_testes, inter)
        if not_ok:
            print("A permuta falhou! As interações não são suficientes para a execução")
            break

        quantidade = int(arr_quantidades[interA])
        if quantidade <= 0:
            continue

        # Imagens dessa classe que estão atualmente em dir_Testes (a serem devolvidas)
        em_dir_testes = [y for y in aT if y.pastaO == pasta.nome and y.em_teste]
        # Imagens dessa classe que ainda não foram testadas (estão em dir_Treino)
        em_dir_treino_ineditas = [
            y for y in aT
            if y.pastaO == pasta.nome and not y.em_teste and not y.ja_testada
        ]

        if len(em_dir_testes) < quantidade or len(em_dir_treino_ineditas) < quantidade:
            print(
                f"Aviso: não há imagens suficientes para rotacionar a classe {pasta.nome}. "
                f"Solicitado={quantidade}, em_dir_Testes={len(em_dir_testes)}, "
                f"disponíveis_em_dir_Treino={len(em_dir_treino_ineditas)}"
            )
            continue

        sorted_testes = rng.sample(em_dir_testes, quantidade)            # voltam para dir_Treino
        sorted_treino = rng.sample(em_dir_treino_ineditas, quantidade)   # vão para dir_Testes

        for old_test, new_test in zip(sorted_testes, sorted_treino):
            nome_perm = old_test.nomeM
            diretorio_test = old_test.diretorioM
            diretorio_treino = new_test.diretorioM

            # 1. A imagem que estava em dir_Testes volta para sua subpasta de classe em dir_Treino.
            destino_volta = dir_treino / old_test.pastaO
            shutil.move(str(diretorio_test), destino_volta)
            pNDM = destino_volta / nome_perm                  # caminho com nome modificado
            pNDO = destino_volta / old_test.nomeO             # caminho com nome original
            pNDM.rename(pNDO)

            if resize_divisor:
                resize_by_divisor(pNDO, resize_divisor, restore_original_size)

            classe_origem = new_test.pastaM                   # nome da subpasta da classe da nova imagem

            old_test.diretorioM = pNDO
            old_test.nomeM = old_test.nomeO
            old_test.pastaM = old_test.pastaO
            old_test.em_teste = False

            # 2. Uma nova imagem sai de dir_Treino para dir_Testes e assume o nome usado antes.
            if resize_divisor:
                # Quando o redimensionamento estraga a imagem por recompressão,
                # o código original recopia a imagem original do dataset.
                if diretorio_treino.exists():
                    diretorio_treino.unlink()
                origem_original = amostras / new_test.pastaO / new_test.nomeO
                shutil.copy(origem_original, dir_testes)
            else:
                shutil.move(str(diretorio_treino), dir_testes)

            sNDM = dir_testes / new_test.nomeO                # caminho com nome original
            sNDO = dir_testes / nome_perm                     # caminho com nome modificado
            sNDM.rename(sNDO)

            new_test.diretorioM = sNDO
            new_test.nomeM = nome_perm
            new_test.pastaM = dir_testes.name
            new_test.em_teste = True
            new_test.ja_testada = True


def divInt(num: int, div: int) -> tuple[list[int], bool]:
    """Divide `num` em `div` partes quase iguais, sinalizando falha quando
    `num < div` ou `div == 0`.

    Mantém a mesma assinatura e semântica da função `divInt` do código original.
    """
    if num >= div and div != 0:
        return split_integer(num, div), False
    print("Não foi possível atender a solicitação")
    return [], True


def check_results(aT: Sequence[Imagem], nome_atual: str, nome_id: str) -> str:
    """Compara a predição com a classe esperada e retorna uma string formatada.

    Equivale à função `checkResults` do código original.
    """
    retorno = "\t-\t"
    for x in aT:
        if nome_atual == x.nomeM:
            if x.pastaO == nome_id:
                return f"{retorno}True\t-\t{x.pastaO}\t-\tFace Encontrada {nome_id}\t-\tAcertou"
            return f"{retorno}False\t-\t{x.pastaO}\t-\tFace Encontrada {nome_id}\t-\tErrou"
    return f"{retorno}None\t-\tNone\t-\tFace Nao Encontrada\t-\tErrou"


def export_list_img(aT: Sequence[Imagem], separador: str, nome_arq: Path) -> None:
    """Salva o estado atual das imagens em arquivo TSV.

    Equivale à função `exportListImg` do código original, com cabeçalho.
    """
    nome_arq.parent.mkdir(parents=True, exist_ok=True)
    header = separador.join(
        ["nomeO", "diretorioO", "pastaO", "nomeM", "diretorioM", "pastaM", "em_teste", "ja_testada"]
    )

    with nome_arq.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for obj in aT:
            fh.write(obj.to_tsv_row(separador) + "\n")


def imprime_list_img(aT: Iterable[Imagem], separador: str = "\t") -> None:
    """Imprime no terminal o estado das imagens. Equivale a `imprimeListImg`."""
    for obj in aT:
        print(obj.to_tsv_row(separador))


def imprime_list_pasta(pastaData: Iterable[Pastas], separador: str = "\t") -> None:
    """Imprime no terminal o estado das pastas. Equivale a `imprimeListPasta`."""
    for obj in pastaData:
        print(obj.nome, obj.qtd, obj.qtd_treino, obj.qtd_testes, sep=separador)


def write_prediction_results(rows: list[dict[str, str]], output_file: Path) -> None:
    """Exporta os resultados das predições de uma execução em TSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "expected_label",
        "predicted_label",
        "correct",
        "face_location",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
