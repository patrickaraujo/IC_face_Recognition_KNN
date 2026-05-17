# IC_face_Recognition_KNN

Iniciação Científica — Reconhecimento facial com KNN.

Projeto de reconhecimento facial que utiliza embeddings da biblioteca
`face_recognition` e um classificador KNN do `scikit-learn`.

## Instalação

### Pacotes de sistema (Ubuntu/Debian)

A biblioteca `face_recognition` depende do `dlib`, que precisa ser
compilado a partir do código-fonte. Antes de instalar as dependências
Python, instale os pacotes de sistema necessários:

```bash
sudo apt install build-essential cmake python3-dev
```

### Dependências Python

```bash
pip install -r requirements.txt
```

Isso inclui `face_recognition_models`, o pacote com os **modelos
pré-treinados** que o `face_recognition` utiliza para detectar faces e
calcular embeddings. Ele é instalado diretamente do GitHub porque não está
disponível no PyPI — sem ele, o `face_recognition` falha na primeira
execução com erro de modelo ausente.

## Execução

Antes de rodar pela primeira vez, confirme que a pasta do dataset tem
uma subpasta por pessoa:

```bash
ls gt_db
# deve listar: s01  s02  s03  ...  s50
```

Caso o conteúdo apareça embrulhado em uma pasta extra (`gt_db/gt_db/s01/...`),
ajuste o caminho passado em `--dataset` para apontar para o nível certo.

Comando padrão:

```bash
python main.py --dataset ./gt_db --workdir ./output/face_knn --porc 70 --inter 3
```

A pasta do dataset deve ter uma subpasta por pessoa/classe.

### Principais parâmetros

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `--dataset` | (obrigatório) | Caminho para a pasta com uma subpasta por pessoa. |
| `--workdir` | `output/face_knn` | Onde modelos, logs e imagens anotadas são salvos. |
| `--porc` | `70` | Percentual que **permanece** em `dir_Treino` (treino). O resto vai para `dir_Testes`. |
| `--inter` | `3` | Número de execuções (treino + predição). Entre execuções, há permuta entre treino e teste. |
| `--n-neighbors` | `3` | Valor de `k` no classificador KNN. |
| `--distance-threshold` | `0.6` | Limiar de distância para aceitar uma predição. |
| `--resize-divisor` | `8` | Divisor de redimensionamento (ver "Redimensionamento de imagens"). |
| `--no-resize` | desligado | Quando passado, não aplica nenhuma transformação de tamanho. |
| `--keep-resized` | desligado | Mantém o tamanho reduzido em vez de restaurar o original. |
| `--seed` | `42` | Semente aleatória, para reprodutibilidade da divisão. |

### Redimensionamento de imagens

Por padrão, cada imagem passa por uma redução seguida de restauração do
tamanho original — comportamento herdado do código original. Isso funciona,
na prática, como um **filtro de degradação**: a resolução nominal não muda,
mas a imagem perde qualidade pela recompressão e reinterpolação. É útil
para avaliar a robustez do KNN a perda de qualidade.

Três cenários comuns:

```bash
# Sem redimensionamento (usa as imagens originais como estão)
python main.py --dataset ./gt_db --porc 70 --inter 3 --no-resize

# Reduz para 1/4 (640×480 → 160×120) e mantém reduzido
python main.py --dataset ./gt_db --porc 70 --inter 3 --resize-divisor 4 --keep-resized

# Degradação (comportamento padrão) com divisor 4
python main.py --dataset ./gt_db --porc 70 --inter 3 --resize-divisor 4
```

O código atual aceita apenas redimensionamento **por divisor** (proporcional
ao tamanho original). Não é possível pedir uma resolução fixa em pixels
(por exemplo, 224×224) sem alterar o código.

### Combinação `--porc` × `--inter`

A permuta entre execuções evita repetir imagens já testadas. Com 15 imagens
por pessoa no Georgia Tech:

| Comando | Por pessoa: treino / teste | Cobertura ao final das `inter` execuções |
|---|---|---|
| `--porc 70 --inter 3` | 10 / 5 | 15 = todas as imagens testadas uma vez |
| `--porc 80 --inter 5` | 12 / 3 | 15 = idem |
| `--porc 50 --inter 2` | 7 / 8 | 15 ou 16 (sobra) |
| `--porc 70 --inter 1` | 10 / 5 | só 5 por pessoa (sem permuta) |

Se `--inter` for maior do que o necessário para esgotar o dataset, o código
emite avisos durante a permuta ("não há imagens suficientes para rotacionar")
e segue em frente, sem realizar trocas inúteis.

## Solução de problemas

### `ModuleNotFoundError: No module named 'pkg_resources'`

A partir da versão **82.0**, o `setuptools` deixou de incluir o módulo
`pkg_resources`, que o `face_recognition_models` ainda usa. Por isso o
`requirements.txt` fixa `setuptools<82`.

Se você instalou os pacotes antes dessa restrição entrar (ou tem um
`setuptools` recente no ambiente), force a reinstalação para uma versão
compatível:

```bash
python -m pip install "setuptools<82" --force-reinstall
```

Em seguida, rode novamente `pip install -r requirements.txt` para garantir
que o `face_recognition_models` esteja consistente.

### `PermissionError: [Errno 13] Permission denied: 'output/...'`

Acontece quando a pasta `output/` foi criada antes com `sudo` (ou ficou
com dono `root`) e agora o script, rodando como usuário comum, não
consegue sobrescrever os arquivos durante o redimensionamento.

Dentro da pasta do projeto:

```bash
sudo chown -R $USER:$USER output
chmod -R u+rwX output
```

Como regra geral, **não rode `main.py` com `sudo`** — não há necessidade,
e isso é justamente o que cria esse problema nas próximas execuções.

## Dataset

Este projeto foi desenvolvido usando o **Georgia Tech Face Database**, mantido
por Ara V. Nefian e disponibilizado pelo Center for Signal and Image
Processing do Georgia Institute of Technology.

Características principais:

- 50 pessoas;
- 15 imagens JPEG coloridas por pessoa, totalizando 750 imagens;
- resolução de 640×480 pixels, com tamanho médio de face de 150×150 pixels;
- fundo não uniforme (cluttered background);
- variação de expressão facial, iluminação, escala e leve rotação da cabeça
  (frontal e/ou inclinada);
- imagens capturadas em duas ou três sessões entre 06/1999 e 11/1999.

### Downloads (página oficial)

- Página do dataset: <http://www.anefian.com/research/face_reco.htm>
- Imagens originais (128 MB): <http://www.anefian.com/research/gt_db.zip>
- Imagens com background removido (15,9 MB): <http://www.anefian.com/research/GTdb_crop.zip>
- Arquivos de rótulo (posição da face): <http://www.anefian.com/research/labels_gt.zip>
- README oficial: <http://www.anefian.com/research/GTDB_README.txt>

### Preparação para uso no projeto

Após o download, basta apontar `--dataset` para a pasta extraída. O dataset
já vem com uma subpasta por pessoa (`s01`, `s02`, ..., `s50`), que é o
formato esperado pelo `main.py`. Com 15 imagens por pessoa e `--porc 70`,
cada classe fica com aproximadamente 10 imagens em `dir_Treino` e 5 em
`dir_Testes`.

### Citação

```bibtex
@misc{nefian_gt_face_db,
  title  = {Georgia Tech face database},
  author = {Nefian, Ara V.},
  url    = {http://www.anefian.com/research/face_reco.htm},
  note   = {Center for Signal and Image Processing,
            Georgia Institute of Technology}
}
```

O uso do dataset é destinado a fins de pesquisa. Consulte a página oficial
para informações de contato e referências de trabalhos que utilizaram a
base.

## Convenção das pastas

- `dir_Treino`: imagens que **alimentam o treino** do classificador KNN.
- `dir_Testes`: imagens usadas para **teste/predição**.
- `--porc`: percentual de imagens que **permanece** em `dir_Treino` (treino);
  o restante é movido para `dir_Testes` (teste).

## Saídas do experimento

Ao terminar, a `--workdir` conterá os modelos KNN treinados, as imagens
de teste anotadas (com caixas e nomes) e a pasta `logs/`. Os arquivos
gerados são:

- `run_info.txt`: parâmetros do experimento, ambiente, caminho absoluto
  do dataset, hora de início, hora de fim, duração total e tempo de
  **cada execução individual**. O mesmo conteúdo é impresso no console
  como banner no início e rodapé no final;
- `predictions_run_N.txt`: predição de cada imagem testada;
- `metrics_run_N_summary.txt`: resumo legível com acurácia, erros de
  identificação, rejeições, falhas de detecção, classes com pior recall
  e pares mais confundidos. **O mesmo conteúdo é impresso no console**
  ao final de cada execução;
- `metrics_run_N_per_class.tsv`: precision, recall, F1 e contagens TP/FN/FP
  por pessoa.

Ao final de todas as execuções é gerado `metrics_overall.tsv` com uma
linha por execução e uma linha agregada, útil para comparar a
estabilidade do classificador entre rodadas.

Detalhes sobre como cada métrica é definida e por que `unknown` e
falhas de detecção são tratadas separadamente estão em
`DOCUMENTACAO_CODIGO.md`.

Consulte `DOCUMENTACAO_CODIGO.md` para detalhes sobre o fluxo do experimento,
os parâmetros disponíveis e as estruturas de dados.
