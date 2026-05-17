"""
Orquestra o experimento completo de reconhecimento facial com KNN.

Fluxo geral:

1. Opcionalmente cria uma cópia renomeada do dataset.
2. Copia todas as imagens para `dir_Treino`, mantendo subpastas por classe.
3. Move `(100 - porc)%` das imagens de cada classe de `dir_Treino` para `dir_Testes`.
4. Treina o classificador KNN com as imagens que permaneceram em `dir_Treino`.
5. Prediz as imagens em `dir_Testes`.
6. Salva imagens anotadas e logs.
7. Se houver mais de uma execução, permuta imagens entre treino e teste.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from dataset_utils import (
    check_results,
    cria_dir_testes,
    cria_dir_treino,
    ensure_empty_dir,
    export_list_img,
    imprime_list_img,
    imprime_list_pasta,
    is_image_file,
    pasta_info,
    permuta,
    renomeia_dataset,
    write_prediction_results,
)
from face_knn import (
    MODEL_FILENAME,
    predict,
    show_prediction_labels_on_image,
    train,
)
from metrics import (
    RunMetrics,
    compute_metrics,
    format_console_summary,
    write_metrics_files,
    write_overall_summary,
)
from models import Imagem
from run_report import RunReport


@dataclass
class ExperimentConfig:
    """Configurações principais do experimento."""

    dataset_dir: Path
    work_dir: Path = Path("output/face_knn")
    porc: int = 70                  # percentual que permanece em dir_Treino
    inter: int = 3                  # número de execuções
    n_neighbors: int = 3
    distance_threshold: float = 0.6
    erase: bool = True
    rename_input: bool = True
    resize_images: bool = True
    resize_divisor: int = 8
    restore_original_size: bool = True
    random_seed: int | None = 42
    verbose: bool = False
    print_state: bool = False
    separador: str = "\t"
    dir_treino_name: str = "dir_Treino"
    dir_testes_name: str = "dir_Testes"


def _format_location(location: tuple[int, int, int, int]) -> str:
    top, right, bottom, left = location
    return f"top={top}, right={right}, bottom={bottom}, left={left}"


def look_for_faces(
    cam_img: Path,
    folder: Path,
    aT: list[Imagem],
    classifier,
    distance_threshold: float,
) -> list[dict[str, str]]:
    """Procura faces em todas as imagens da pasta de teste.

    Equivale à função `lookF_Faces` do código original, e retorna também as
    linhas estruturadas para serem exportadas em log TSV.
    """
    rows: list[dict[str, str]] = []

    for image_path in sorted(cam_img.iterdir()):
        if not is_image_file(image_path):
            continue

        print(f"Looking for faces in {image_path.name}", end="")

        predictions = predict(
            image_path,
            knn_clf=classifier,
            distance_threshold=distance_threshold,
        )

        for name, (top, right, bottom, left) in predictions:
            resultado = check_results(aT, image_path.name, name)
            print(f"\t-\tFound {name} at ({left}, {top}){resultado}", end="")

            # Determina a classe esperada para o log estruturado
            expected_label = next(
                (x.pastaO for x in aT if x.nomeM == image_path.name),
                "unknown_expected",
            )
            rows.append({
                "image_name": image_path.name,
                "expected_label": expected_label,
                "predicted_label": name,
                "correct": str(name == expected_label),
                "face_location": _format_location((top, right, bottom, left)),
            })

        if not predictions:
            expected_label = next(
                (x.pastaO for x in aT if x.nomeM == image_path.name),
                "unknown_expected",
            )
            print(f"\t-\tNenhuma face encontrada\t-\tEsperado {expected_label}", end="")
            rows.append({
                "image_name": image_path.name,
                "expected_label": expected_label,
                "predicted_label": "no_face_detected",
                "correct": "False",
                "face_location": "",
            })

        print()

        # Salva imagem anotada
        show_prediction_labels_on_image(image_path, predictions, folder / image_path.name)

    return rows


def run_experiment(config: ExperimentConfig) -> None:
    """Executa o pipeline completo."""
    started_at = datetime.now()
    rng = random.Random(config.random_seed)

    ensure_empty_dir(config.work_dir, erase=config.erase)

    # 1. Cópia renomeada do dataset (opcional)
    dataset_to_use = config.dataset_dir
    if config.rename_input:
        renamed_dir = config.work_dir / "dataset_renomeado"
        dataset_to_use = renomeia_dataset(config.dataset_dir, renamed_dir, erase=True)

    # Diretórios derivados
    dir_treino = config.work_dir / config.dir_treino_name
    dir_testes = config.work_dir / config.dir_testes_name
    logs_dir = config.work_dir / "logs"
    predictions_dir = config.work_dir / "predictions"
    models_dir = config.work_dir / "models"

    ensure_empty_dir(logs_dir)
    ensure_empty_dir(predictions_dir)
    ensure_empty_dir(models_dir)

    # Imprime e grava o cabeçalho com configurações e hora de início.
    # O arquivo será reescrito ao final com fim/duração.
    info_inicial = format_run_info(
        config=config,
        started_at=started_at,
        dataset_in_use=dataset_to_use,
    )
    print(info_inicial)
    write_run_info(info_inicial, logs_dir)

    # 2. Cria dir_Treino com todas as imagens (alimentará o treino do KNN)
    aT = cria_dir_treino(
        dataset_to_use,
        dir_treino,
        erase=True,
        resize_divisor=config.resize_divisor if config.resize_images else None,
        restore_original_size=config.restore_original_size,
    )

    print("\nConfiguração inicial das pastas")
    if config.print_state:
        imprime_list_img(aT, config.separador)

    # 3. Move (100 - porc)% das imagens de dir_Treino para dir_Testes
    aT = cria_dir_testes(
        dir_treino=dir_treino,
        porc=config.porc,
        dir_testes=dir_testes,
        nomeT="teste",
        aT=aT,
        rng=rng,
        erase=True,
    )

    print("\nConfiguração das pastas após a criação do diretório de testes")
    if config.print_state:
        imprime_list_img(aT, config.separador)
    export_list_img(aT, config.separador, logs_dir / "log_run_1.txt")

    # 4. Loop principal de execuções
    all_run_metrics: list[RunMetrics] = []
    for x in range(config.inter):
        print(f"\nExecução:\t{x + 1}")
        print("Training KNN classifier...", end="\t")

        model_path = models_dir / f"run_{x + 1}_{MODEL_FILENAME}"
        classifier = train(
            dir_treino,
            model_save_path=model_path,
            n_neighbors=config.n_neighbors,
            verbose=config.verbose,
        )
        print("Training complete!")

        # Salva imagens anotadas em uma subpasta por execução, se houver mais de uma
        folder = predictions_dir / f"run_{x + 1}" if config.inter > 1 else predictions_dir / "imgs"
        ensure_empty_dir(folder, erase=True)

        rows = look_for_faces(
            cam_img=dir_testes,
            folder=folder,
            aT=aT,
            classifier=classifier,
            distance_threshold=config.distance_threshold,
        )
        write_prediction_results(rows, logs_dir / f"predictions_run_{x + 1}.txt")

        # Calcula e exibe as métricas desta execução
        run_metrics = compute_metrics(rows, run_number=x + 1)
        print(format_console_summary(run_metrics))
        write_metrics_files(run_metrics, logs_dir)
        all_run_metrics.append(run_metrics)

        # Estatísticas por classe
        stats = pasta_info(aT)
        if config.print_state:
            print("\nEstatísticas por classe (nome, qtd, em_treino, em_testes):")
            imprime_list_pasta(stats, config.separador)

        # 5. Permuta imagens entre os conjuntos para a próxima execução
        if x < config.inter - 1:
            print("Permutando imagens entre dir_Treino e dir_Testes...")
            permuta(
                aT=aT,
                pastasL=stats,
                inter=config.inter - 1,
                interA=x,
                rng=rng,
                dir_treino=dir_treino,
                dir_testes=dir_testes,
                amostras=dataset_to_use,
                resize_divisor=config.resize_divisor if config.resize_images else None,
                restore_original_size=config.restore_original_size,
            )
            print(f"\nConfiguração das pastas após a permuta {x + 1}")
            if config.print_state:
                imprime_list_img(aT, config.separador)
            export_list_img(aT, config.separador, logs_dir / f"log_run_{x + 2}.txt")

    # 6. Resumo agregado de todas as execuções
    if all_run_metrics:
        write_overall_summary(all_run_metrics, logs_dir)

        total_geral = sum(m.global_metrics.total for m in all_run_metrics)
        acertos_geral = sum(m.global_metrics.acertos for m in all_run_metrics)
        if total_geral:
            print("")
            print("=== Resumo agregado de todas as execuções ===")
            print(f"Total de predições:  {total_geral}")
            print(
                f"Acertos:             {acertos_geral}  "
                f"({(acertos_geral / total_geral) * 100:.2f}%)"
            )

    # 7. Atualiza run_info.txt com fim e duração; imprime resumo de tempo
    finished_at = datetime.now()
    info_final = format_run_info(
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        dataset_in_use=dataset_to_use,
    )
    write_run_info(info_final, logs_dir)
    print("")
    print(info_final)

    print(f"\nExperimento finalizado. Resultados em: {config.work_dir}")
