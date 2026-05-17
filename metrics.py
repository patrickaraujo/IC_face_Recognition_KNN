"""
Cálculo e formatação de métricas de avaliação a partir das predições.

A entrada é a lista de linhas (`rows`) produzida por `look_for_faces` em
`experiment.py`. Cada linha tem:

- `image_name`: nome do arquivo testado;
- `expected_label`: classe verdadeira (ex.: `s01`);
- `predicted_label`: classe predita pelo KNN, ou `unknown` se a distância
  ficou acima do limiar, ou `no_face_detected` se nenhuma face foi
  encontrada na imagem;
- `correct`: string "True"/"False";
- `face_location`: bounding box, quando aplicável.

Definições adotadas
-------------------
No dataset Georgia Tech todas as imagens de teste contêm pessoas que
também aparecem no treino, então não há "verdadeiro negativo" no sentido
binário usual. As métricas são calculadas no contexto multiclasse com
classe especial `unknown`:

- **Acerto**: `predicted == expected`. Equivale a TP para essa classe.
- **Erro de identificação**: o KNN respondeu uma pessoa, mas a errada.
  Equivale a FP para a classe predita e FN para a classe esperada.
- **Rejeição**: o KNN respondeu `unknown` (distância acima do limiar).
  Equivale a FN para a classe esperada.
- **Falha de detecção**: a biblioteca não encontrou face na imagem.
  Conta como erro mas é uma categoria separada porque não envolve KNN.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

UNKNOWN = "unknown"
NO_FACE = "no_face_detected"


@dataclass
class GlobalMetrics:
    """Resumo global das predições de uma execução."""

    total: int = 0
    acertos: int = 0
    erros_identificacao: int = 0
    rejeicoes_unknown: int = 0
    falhas_deteccao: int = 0

    @property
    def acuracia(self) -> float:
        return self.acertos / self.total if self.total else 0.0

    @property
    def taxa_erro_identificacao(self) -> float:
        return self.erros_identificacao / self.total if self.total else 0.0

    @property
    def taxa_rejeicao(self) -> float:
        return self.rejeicoes_unknown / self.total if self.total else 0.0

    @property
    def taxa_falha_deteccao(self) -> float:
        return self.falhas_deteccao / self.total if self.total else 0.0


@dataclass
class ClassMetrics:
    """Métricas para uma classe específica (uma pessoa)."""

    label: str
    suporte: int = 0        # quantas imagens dessa classe foram testadas
    tp: int = 0             # acertos da classe (expected = predicted = label)
    fn: int = 0             # imagens da classe preditas como outra pessoa ou unknown ou no_face
    fp: int = 0             # imagens de outra classe preditas como esta classe

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0


@dataclass
class RunMetrics:
    """Resultado completo das métricas de uma execução."""

    run_number: int
    global_metrics: GlobalMetrics
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion_pairs: dict[tuple[str, str], int] = field(default_factory=dict)


def compute_metrics(rows: list[dict[str, str]], run_number: int) -> RunMetrics:
    """Computa métricas globais, por classe e pares de confusão."""
    global_m = GlobalMetrics()
    per_class: dict[str, ClassMetrics] = {}
    confusion: dict[tuple[str, str], int] = {}

    # Cada linha de `rows` é uma predição. Imagens com múltiplas faces
    # geram múltiplas linhas — todas são contabilizadas.
    for row in rows:
        expected = row["expected_label"]
        predicted = row["predicted_label"]

        # Garante que cada classe vista (esperada ou predita, exceto especiais)
        # tenha uma entrada em per_class.
        if expected not in per_class:
            per_class[expected] = ClassMetrics(label=expected)
        if predicted not in (UNKNOWN, NO_FACE) and predicted not in per_class:
            per_class[predicted] = ClassMetrics(label=predicted)

        per_class[expected].suporte += 1
        global_m.total += 1

        if predicted == NO_FACE:
            global_m.falhas_deteccao += 1
            per_class[expected].fn += 1
        elif predicted == UNKNOWN:
            global_m.rejeicoes_unknown += 1
            per_class[expected].fn += 1
        elif predicted == expected:
            global_m.acertos += 1
            per_class[expected].tp += 1
        else:
            global_m.erros_identificacao += 1
            per_class[expected].fn += 1
            per_class[predicted].fp += 1
            pair = (expected, predicted)
            confusion[pair] = confusion.get(pair, 0) + 1

    return RunMetrics(
        run_number=run_number,
        global_metrics=global_m,
        per_class=per_class,
        confusion_pairs=confusion,
    )


def format_console_summary(metrics: RunMetrics, top_n_confusions: int = 5,
                           worst_n_classes: int = 5) -> str:
    """Monta um resumo enxuto para imprimir no console.

    Mostra o resumo global e destaca as classes com pior recall e os pares
    de confusão mais frequentes. Não imprime a tabela completa por classe
    no terminal — essa fica no log TSV.
    """
    g = metrics.global_metrics
    lines: list[str] = []

    lines.append("")
    lines.append(f"=== Métricas da execução {metrics.run_number} ===")
    lines.append(f"Total de predições:       {g.total}")
    lines.append(
        f"Acertos:                  {g.acertos:>5}  ({g.acuracia * 100:6.2f}%)"
    )
    lines.append(
        f"Erros de identificação:   {g.erros_identificacao:>5}  "
        f"({g.taxa_erro_identificacao * 100:6.2f}%)"
    )
    lines.append(
        f"Rejeições (unknown):      {g.rejeicoes_unknown:>5}  "
        f"({g.taxa_rejeicao * 100:6.2f}%)"
    )
    lines.append(
        f"Falhas de detecção:       {g.falhas_deteccao:>5}  "
        f"({g.taxa_falha_deteccao * 100:6.2f}%)"
    )

    # Classes que mais erraram, ordenadas por recall crescente
    classes_com_suporte = [c for c in metrics.per_class.values() if c.suporte > 0]
    piores = sorted(classes_com_suporte, key=lambda c: (c.recall, c.label))[:worst_n_classes]
    if piores:
        lines.append("")
        lines.append(f"Classes com pior recall (top {len(piores)}):")
        lines.append("  classe   suporte    tp    fn    fp   prec    rec     f1")
        for c in piores:
            lines.append(
                f"  {c.label:<8} {c.suporte:>5}    {c.tp:>3}   {c.fn:>3}   {c.fp:>3}  "
                f"{c.precision * 100:5.1f}  {c.recall * 100:5.1f}  {c.f1 * 100:5.1f}"
            )

    # Pares de confusão mais frequentes
    if metrics.confusion_pairs:
        top_pairs = sorted(
            metrics.confusion_pairs.items(), key=lambda kv: (-kv[1], kv[0])
        )[:top_n_confusions]
        lines.append("")
        lines.append(f"Pares mais confundidos (top {len(top_pairs)}):")
        for (esperado, predito), count in top_pairs:
            lines.append(f"  {esperado} -> {predito}: {count}")

    return "\n".join(lines)


def write_metrics_files(metrics: RunMetrics, logs_dir: Path) -> tuple[Path, Path]:
    """Grava dois arquivos: resumo legível em TXT e métricas por classe em TSV.

    Retorna `(caminho_resumo_txt, caminho_per_class_tsv)`.
    """
    run = metrics.run_number
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resumo legível
    summary_path = logs_dir / f"metrics_run_{run}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(format_console_summary(metrics))
        fh.write("\n")

    # 2. Métricas por classe em TSV (para análise posterior)
    per_class_path = logs_dir / f"metrics_run_{run}_per_class.tsv"
    with per_class_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "label", "suporte", "tp", "fn", "fp",
            "precision", "recall", "f1",
        ])
        for label in sorted(metrics.per_class.keys()):
            c = metrics.per_class[label]
            writer.writerow([
                c.label,
                c.suporte,
                c.tp, c.fn, c.fp,
                f"{c.precision:.4f}",
                f"{c.recall:.4f}",
                f"{c.f1:.4f}",
            ])

    return summary_path, per_class_path


def write_overall_summary(all_runs: list[RunMetrics], logs_dir: Path) -> Path:
    """Grava um resumo agregando todas as execuções num único arquivo."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "metrics_overall.tsv"

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "run", "total", "acertos", "erros_identificacao",
            "rejeicoes_unknown", "falhas_deteccao",
            "acuracia", "taxa_erro_identificacao",
            "taxa_rejeicao", "taxa_falha_deteccao",
        ])

        total_geral = 0
        acertos_geral = 0
        for m in all_runs:
            g = m.global_metrics
            writer.writerow([
                m.run_number, g.total, g.acertos, g.erros_identificacao,
                g.rejeicoes_unknown, g.falhas_deteccao,
                f"{g.acuracia:.4f}",
                f"{g.taxa_erro_identificacao:.4f}",
                f"{g.taxa_rejeicao:.4f}",
                f"{g.taxa_falha_deteccao:.4f}",
            ])
            total_geral += g.total
            acertos_geral += g.acertos

        # Linha agregada de todas as runs
        if total_geral:
            acuracia_total = acertos_geral / total_geral
            writer.writerow([
                "all", total_geral, acertos_geral,
                sum(m.global_metrics.erros_identificacao for m in all_runs),
                sum(m.global_metrics.rejeicoes_unknown for m in all_runs),
                sum(m.global_metrics.falhas_deteccao for m in all_runs),
                f"{acuracia_total:.4f}",
                "", "", "",
            ])

    return path


def _format_duration(delta: timedelta) -> str:
    """Formata uma duração em horas, minutos e segundos."""
    total_seconds = int(delta.total_seconds())
    horas, resto = divmod(total_seconds, 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas}h {minutos:02d}m {segundos:02d}s"
    if minutos:
        return f"{minutos}m {segundos:02d}s"
    return f"{segundos}s"


def format_run_info(
    config,
    started_at: datetime,
    finished_at: datetime | None = None,
    dataset_in_use: Path | None = None,
) -> str:
    """Monta o cabeçalho informativo da execução.

    Inclui caminhos, parâmetros principais, horários e duração total.
    O mesmo texto é usado para impressão no console e para o arquivo
    `run_info.txt`.

    Parâmetros
    ----------
    config:
        Objeto `ExperimentConfig` (ou compatível) com os campos do experimento.
    started_at:
        Datetime do início da execução.
    finished_at:
        Datetime do fim. Quando None, apenas o cabeçalho inicial é gerado
        (sem a linha de fim e duração).
    dataset_in_use:
        Caminho efetivamente usado pelo treino (pode ser o dataset_renomeado,
        que difere do `config.dataset_dir`).
    """
    iso = "%Y-%m-%d %H:%M:%S"

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Configurações do experimento")
    lines.append("=" * 60)

    # Caminhos
    lines.append("")
    lines.append("Caminhos:")
    lines.append(f"  --dataset           {config.dataset_dir}")
    lines.append(f"  --workdir           {config.work_dir}")
    if dataset_in_use is not None and Path(dataset_in_use) != Path(config.dataset_dir):
        lines.append(f"  dataset em uso      {dataset_in_use}")
    lines.append(f"  dir_Treino          {config.work_dir / config.dir_treino_name}")
    lines.append(f"  dir_Testes          {config.work_dir / config.dir_testes_name}")

    # Parâmetros do experimento
    lines.append("")
    lines.append("Parâmetros:")
    lines.append(f"  --porc              {config.porc} (% que permanece em dir_Treino)")
    lines.append(f"  --inter             {config.inter} (execuções)")
    lines.append(f"  --n-neighbors       {config.n_neighbors} (k do KNN)")
    lines.append(f"  --distance-threshold {config.distance_threshold}")
    lines.append(f"  --seed              {config.random_seed}")

    # Redimensionamento
    lines.append("")
    lines.append("Redimensionamento:")
    if not config.resize_images:
        lines.append("  desativado (--no-resize)")
    else:
        modo = ("reduz e mantém reduzido"
                if not config.restore_original_size
                else "reduz e restaura tamanho original (filtro de degradação)")
        lines.append(f"  divisor: {config.resize_divisor} ({modo})")

    # Outras opções
    lines.append("")
    lines.append("Opções:")
    lines.append(f"  apagar saída antiga          {config.erase}")
    lines.append(f"  renomear dataset de entrada  {config.rename_input}")

    # Tempos
    lines.append("")
    lines.append("Tempos:")
    lines.append(f"  Início:   {started_at.strftime(iso)}")
    if finished_at is not None:
        duracao = finished_at - started_at
        lines.append(f"  Fim:      {finished_at.strftime(iso)}")
        lines.append(f"  Duração:  {_format_duration(duracao)}")

    lines.append("=" * 60)
    return "\n".join(lines)


def write_run_info(text: str, logs_dir: Path) -> Path:
    """Grava o cabeçalho informativo em `run_info.txt`."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "run_info.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path
