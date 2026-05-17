"""
Relatório de execução: configurações, tempos e localização do dataset.

Este módulo é responsável por imprimir o banner inicial (com os
parâmetros usados), registrar o tempo de cada execução, e gerar o
arquivo `logs/run_info.txt` com o relatório completo da rodada.

A intenção é separar o "metadado de execução" (parâmetros, tempo,
sistema) das métricas de modelo (que ficam em `metrics.py`).
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment import ExperimentConfig

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_seconds(seconds: float) -> str:
    """Formata duração em hh:mm:ss.ms para leitura rápida."""
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _format_resize_behavior(config: "ExperimentConfig") -> str:
    """Descreve em texto curto o que o experimento faz com o tamanho das imagens."""
    if not config.resize_images:
        return "desativado"
    if config.restore_original_size:
        return (
            f"redução por divisor {config.resize_divisor} com restauração "
            f"do tamanho original (filtro de degradação)"
        )
    return f"redução permanente por divisor {config.resize_divisor}"


def build_config_lines(config: "ExperimentConfig", dataset_abs: Path) -> list[str]:
    """Monta as linhas do bloco de configuração (sem cabeçalho)."""
    return [
        f"Dataset (caminho absoluto):  {dataset_abs}",
        f"Workdir:                     {config.work_dir.resolve()}",
        f"porc (% em dir_Treino):      {config.porc}",
        f"inter (execuções):           {config.inter}",
        f"n_neighbors (k do KNN):      {config.n_neighbors}",
        f"distance_threshold:          {config.distance_threshold}",
        f"redimensionamento:           {_format_resize_behavior(config)}",
        f"renomear dataset (prefixar): {config.rename_input}",
        f"random_seed:                 {config.random_seed}",
    ]


@dataclass
class RunTiming:
    """Duração e contagem de imagens de uma execução individual."""

    run_number: int
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    images_evaluated: int = 0

    @property
    def duration_pretty(self) -> str:
        return _format_seconds(self.duration_seconds)


@dataclass
class RunReport:
    """Coleta tempos e parâmetros do experimento inteiro.

    Uso típico em `experiment.py`:

        report = RunReport.start(config)
        report.print_header()
        ...
        with report.time_run(run_number=N) as t:
            ... treina, prediz ...
            t.images_evaluated = len(rows)
        ...
        report.finish()
        report.print_footer()
        report.write_to_file(logs_dir)
    """

    config: "ExperimentConfig"
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    total_duration_seconds: float | None = None
    run_timings: list[RunTiming] = field(default_factory=list)
    dataset_used: Path | None = None
    _start_perf: float = field(default_factory=perf_counter)

    @classmethod
    def start(cls, config: "ExperimentConfig") -> "RunReport":
        """Cria um relatório com hora de início registrada agora."""
        return cls(config=config)

    def set_dataset_used(self, dataset_used: Path) -> None:
        """Registra qual dataset foi efetivamente lido (ex.: o renomeado)."""
        self.dataset_used = dataset_used

    def finish(self) -> None:
        """Marca o término do experimento."""
        self.finished_at = datetime.now()
        self.total_duration_seconds = perf_counter() - self._start_perf

    def print_header(self) -> None:
        """Imprime o banner inicial com configurações e hora de início."""
        dataset_abs = self.config.dataset_dir.resolve()

        print("")
        print("=" * 60)
        print("Experimento de reconhecimento facial com KNN")
        print("=" * 60)
        print(f"Início:                      {self.started_at.strftime(DATE_FORMAT)}")
        for line in build_config_lines(self.config, dataset_abs):
            print(line)
        print("=" * 60)

    def print_footer(self) -> None:
        """Imprime o resumo final com tempos por execução e tempo total."""
        if self.finished_at is None or self.total_duration_seconds is None:
            return

        print("")
        print("=" * 60)
        print("Tempos de execução")
        print("=" * 60)
        for t in self.run_timings:
            print(
                f"Run {t.run_number}: "
                f"início {t.started_at.strftime(DATE_FORMAT)}  "
                f"fim {t.finished_at.strftime(DATE_FORMAT)}  "
                f"duração {t.duration_pretty}  "
                f"imagens={t.images_evaluated}"
            )
        print("-" * 60)
        print(f"Início do experimento:  {self.started_at.strftime(DATE_FORMAT)}")
        print(f"Término do experimento: {self.finished_at.strftime(DATE_FORMAT)}")
        print(f"Duração total:          {_format_seconds(self.total_duration_seconds)}")
        print("=" * 60)

    class _RunTimerContext:
        """Context manager retornado por `time_run`."""

        def __init__(self, report: "RunReport", run_number: int) -> None:
            self._report = report
            self.timing = RunTiming(
                run_number=run_number,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                duration_seconds=0.0,
            )
            self._start_perf = 0.0

        def __enter__(self) -> RunTiming:
            self.timing.started_at = datetime.now()
            self._start_perf = perf_counter()
            return self.timing

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self.timing.finished_at = datetime.now()
            self.timing.duration_seconds = perf_counter() - self._start_perf
            self._report.run_timings.append(self.timing)

    def time_run(self, run_number: int) -> "_RunTimerContext":
        """Context manager que mede uma execução individual."""
        return self._RunTimerContext(self, run_number)

    def write_to_file(self, logs_dir: Path) -> Path:
        """Grava o relatório completo no arquivo `logs/run_info.txt`."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / "run_info.txt"

        dataset_abs = self.config.dataset_dir.resolve()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Relatório de execução do experimento")
        lines.append("=" * 60)
        lines.append("")
        lines.append("[Configurações]")
        for line in build_config_lines(self.config, dataset_abs):
            lines.append(line)
        if self.dataset_used and self.dataset_used.resolve() != dataset_abs:
            lines.append(
                f"Dataset efetivamente lido:   {self.dataset_used.resolve()}"
            )
        lines.append("")

        lines.append("[Ambiente]")
        lines.append(f"Python:  {sys.version.split()[0]} ({platform.python_implementation()})")
        lines.append(f"Sistema: {platform.system()} {platform.release()}")
        lines.append(f"Máquina: {platform.machine()}")
        lines.append("")

        lines.append("[Tempos]")
        lines.append(f"Início do experimento:  {self.started_at.strftime(DATE_FORMAT)}")
        if self.finished_at:
            lines.append(f"Término do experimento: {self.finished_at.strftime(DATE_FORMAT)}")
        if self.total_duration_seconds is not None:
            lines.append(f"Duração total:          {_format_seconds(self.total_duration_seconds)}")
        lines.append("")

        if self.run_timings:
            lines.append("[Tempos por execução]")
            lines.append("run\tinicio\tfim\tduracao\timagens_avaliadas")
            for t in self.run_timings:
                lines.append(
                    f"{t.run_number}\t"
                    f"{t.started_at.strftime(DATE_FORMAT)}\t"
                    f"{t.finished_at.strftime(DATE_FORMAT)}\t"
                    f"{t.duration_pretty}\t"
                    f"{t.images_evaluated}"
                )

        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            fh.write("\n")

        return path
