"""
Ponto de entrada do projeto de reconhecimento facial com KNN.

Exemplo de uso:

    python main.py --dataset ./gt_db --workdir ./output/face_knn --porc 70 --inter 3

A pasta do dataset deve ter uma subpasta por pessoa/classe.

Convenção das pastas:
- `dir_Treino`: alimenta o **treino** do classificador KNN.
- `dir_Testes`: imagens usadas para **teste/predição**.
- `--porc`: percentual de imagens que **permanece** em `dir_Treino`
  (o restante vai para `dir_Testes`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Reconhecimento facial com KNN usando a biblioteca face_recognition."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Caminho para o dataset original. Deve conter uma subpasta por pessoa.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("output/face_knn"),
        help="Pasta onde serão salvos arquivos temporários, modelos, logs e imagens anotadas.",
    )
    parser.add_argument(
        "--porc",
        type=int,
        default=70,
        help=(
            "Percentual de imagens de cada classe que permanece em dir_Treino. "
            "O restante é movido para dir_Testes."
        ),
    )
    parser.add_argument(
        "--inter",
        type=int,
        default=3,
        help="Número de execuções. Quando maior que 1, permuta imagens entre os conjuntos.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="Valor de k usado pelo classificador KNN.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.6,
        help="Limiar de distância para aceitar uma predição. Valores maiores aceitam mais correspondências.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para tornar a divisão reprodutível.",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="Não cria a cópia renomeada do dataset.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Não aplica redimensionamento nas imagens.",
    )
    parser.add_argument(
        "--resize-divisor",
        type=int,
        default=8,
        help="Divisor usado para reduzir temporariamente ou permanentemente o tamanho das imagens.",
    )
    parser.add_argument(
        "--keep-resized",
        action="store_true",
        help="Mantém as imagens reduzidas. Sem esta opção, elas voltam ao tamanho original após a redução.",
    )
    parser.add_argument(
        "--keep-old-output",
        action="store_true",
        help="Não apaga a pasta de saída antes de executar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra mensagens extras durante o treinamento.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Imprime no terminal o estado das imagens em cada etapa.",
    )

    return parser.parse_args()


def main() -> None:
    """Monta a configuração e executa o experimento."""
    args = parse_args()

    config = ExperimentConfig(
        dataset_dir=args.dataset,
        work_dir=args.workdir,
        porc=args.porc,
        inter=args.inter,
        n_neighbors=args.n_neighbors,
        distance_threshold=args.distance_threshold,
        erase=not args.keep_old_output,
        rename_input=not args.no_rename,
        resize_images=not args.no_resize,
        resize_divisor=args.resize_divisor,
        restore_original_size=not args.keep_resized,
        random_seed=args.seed,
        verbose=args.verbose,
        print_state=args.print_state,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
