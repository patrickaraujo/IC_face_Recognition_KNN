"""
Modelos de dados usados pelo experimento de reconhecimento facial.

Este arquivo substitui as classes simples `Imagem` e `Pastas` do código
original por dataclasses. A semântica das pastas segue a convenção usual
de aprendizado de máquina:

- `dir_Treino`: pasta que alimenta o **treino** do classificador KNN.
- `dir_Testes`: pasta com as imagens usadas para **teste/predição**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Imagem:
    """Representa uma imagem ao longo do experimento.

    A mesma imagem pode mudar de nome e de pasta durante a execução:
    inicialmente ela fica em `dir_Treino` (pool que alimenta o treino do
    KNN), parte das imagens é movida para `dir_Testes` para predição, e
    em execuções posteriores algumas podem voltar para `dir_Treino`.
    """

    nomeO: str          # Nome original do arquivo
    diretorioO: Path    # Caminho original
    pastaO: str         # Pasta/classe original (nome da pessoa)
    nomeM: str          # Nome atual (modificado)
    diretorioM: Path    # Caminho atual
    pastaM: str         # Pasta atual
    em_teste: bool = False      # True quando está em dir_Testes
    ja_testada: bool = False    # True quando já passou por dir_Testes em alguma execução

    def to_tsv_row(self, separator: str = "\t") -> str:
        """Converte o registro para uma linha de log em formato TSV."""
        values = [
            self.nomeO,
            str(self.diretorioO),
            self.pastaO,
            self.nomeM,
            str(self.diretorioM),
            self.pastaM,
            str(self.em_teste),
            str(self.ja_testada),
        ]
        return separator.join(values)


@dataclass
class Pastas:
    """Resumo de quantidade de imagens por pessoa/classe.

    Atributos:

    - `nome`: nome da pasta/classe;
    - `qtd`: quantidade total de imagens da classe;
    - `qtd_testes`: quantidade de imagens atualmente em `dir_Testes`;
    - `qtd_treino` (propriedade): quantidade em `dir_Treino` (resto).
    """

    nome: str
    qtd: int
    qtd_testes: int

    @property
    def qtd_treino(self) -> int:
        return self.qtd - self.qtd_testes
