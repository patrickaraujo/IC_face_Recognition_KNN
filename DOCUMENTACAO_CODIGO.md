# Documentação do código: Reconhecimento facial com KNN

## 1. Visão geral

Este projeto implementa um pipeline de reconhecimento facial usando:

- `face_recognition` para detectar faces e extrair embeddings faciais;
- `scikit-learn` para treinar um classificador KNN;
- `Pillow` para salvar imagens anotadas com caixas e nomes preditos;
- arquivos TSV para registrar o estado das imagens e os resultados das predições.

O código original foi reorganizado para separar responsabilidades, remover
caminhos fixos do Windows, reduzir importações globais e deixar o fluxo
mais claro. A nomenclatura das pastas e do parâmetro de divisão foi
ajustada para refletir a convenção usual de aprendizado de máquina.

## 2. Convenção das pastas

O projeto adota a convenção usual de ML:

| Nome | Função no experimento |
|---|---|
| `dir_Treino` | Imagens que **alimentam o treino** do classificador KNN. |
| `dir_Testes` | Imagens usadas para **teste/predição** do modelo treinado. |
| `porc` | Percentual de imagens de cada classe que **permanece em `dir_Treino`** (treino). O restante é movido para `dir_Testes` (teste). |

Por exemplo, ao executar:

```bash
python main.py --dataset ./gt_db --porc 70 --inter 3
```

cerca de 70% das imagens de cada pessoa permanecem em `dir_Treino` e
alimentam o treino do KNN; os outros 30% vão para `dir_Testes` e são
usados para teste/predição.

Em execuções subsequentes (controladas por `--inter`), o código permuta
imagens entre os dois conjuntos, de modo análogo a uma validação cruzada
simples.

## 3. Fluxo do experimento

O pipeline executa as seguintes etapas:

1. Lê o dataset original, que deve ter uma subpasta por pessoa.
2. Opcionalmente cria uma cópia renomeada do dataset, prefixando cada
   imagem com o nome da classe.
3. Copia todas as imagens para `dir_Treino`, mantendo a estrutura de
   subpastas por classe.
4. Move `(100 − porc)%` das imagens de cada classe de `dir_Treino` para
   `dir_Testes`.
5. Treina o classificador KNN com as imagens que permaneceram em
   `dir_Treino`.
6. Prediz as faces das imagens em `dir_Testes`.
7. Salva imagens anotadas e logs com os resultados.
8. Se `--inter` for maior que 1, permuta parte das imagens entre os dois
   conjuntos e repete o ciclo.

## 4. Estrutura dos arquivos

```text
main.py
experiment.py
face_knn.py
dataset_utils.py
metrics.py
models.py
requirements.txt
README.md
DOCUMENTACAO_CODIGO.md
```

### `main.py`

É o ponto de entrada do projeto. Ele usa `argparse` para ler os parâmetros
de execução no terminal.

Responsabilidades principais:

- ler o caminho do dataset;
- ler parâmetros como número de execuções (`--inter`), valor de `k`,
  limiar de distância e percentual `--porc`;
- montar um objeto `ExperimentConfig`;
- chamar `run_experiment(config)`.

### `experiment.py`

Orquestra o experimento completo.

Principais elementos:

- `ExperimentConfig`: guarda todas as configurações do experimento;
- `run_experiment`: executa o pipeline inteiro;
- `look_for_faces`: aplica o classificador treinado nas imagens de teste.

Fluxo dentro de `run_experiment`:

1. cria a pasta de saída;
2. cria a cópia renomeada do dataset, se habilitado;
3. cria `dir_Treino` com todas as imagens (mantém subpasta por classe);
4. move `(100 − porc)%` para `dir_Testes`;
5. treina o KNN com `dir_Treino`;
6. executa predições sobre `dir_Testes`;
7. exporta logs;
8. permuta imagens entre os conjuntos, se `inter > 1`.

### `face_knn.py`

Contém a parte de reconhecimento facial e KNN.

Principais funções:

- `train`: treina o KNN com imagens organizadas por classe;
- `predict`: reconhece faces em uma imagem usando o modelo treinado;
- `show_prediction_labels_on_image`: desenha caixas e rótulos na imagem
  e salva o resultado.

No treinamento, cada imagem precisa ter exatamente uma face. Se uma imagem
não tiver face ou tiver mais de uma, ela é ignorada.

### `dataset_utils.py`

Contém funções auxiliares para manipulação de pastas, imagens e logs.
Os nomes das principais funções preservam a correspondência com o código
original:

| Função atual | Equivalente no código original |
|---|---|
| `renomeia_dataset` | `novoDirRenomeia` |
| `cria_dir_treino` | `criaDirTestes` (cria o pool inicial completo) |
| `cria_dir_testes` | `criaDirTreinamento` (separa o subconjunto de teste) |
| `permuta` | `permuta` |
| `pasta_info` | `pastaInfo` |
| `check_results` | `checkResults` |
| `divInt` | `divInt` |
| `export_list_img` | `exportListImg` |
| `imprime_list_img` | `imprimeListImg` |
| `imprime_list_pasta` | `imprimeListPasta` |
| `resize_by_divisor` / `resize_image` | `adImg` / `resizeImg` |

Observação: os papéis das duas primeiras funções foram trocados em relação
aos nomes do código original, para que o nome de cada função reflita o
papel real que ela exerce no novo pipeline.

### `models.py`

Define as estruturas de dados do projeto como `dataclasses`
(no lugar das classes manuais `Imagem` e `Pastas` do código original).

- `Imagem`: guarda o nome, caminho e pasta original (`nomeO`, `diretorioO`,
  `pastaO`), o nome, caminho e pasta atuais (`nomeM`, `diretorioM`,
  `pastaM`), e os flags `em_teste` (a imagem está atualmente em
  `dir_Testes`) e `ja_testada` (já passou por `dir_Testes` em alguma
  execução).
- `Pastas`: guarda `nome`, `qtd` (total da classe), `qtd_testes`
  (quantas estão em `dir_Testes`) e a propriedade `qtd_treino` (resto).

## 5. Como o KNN é usado

O KNN não trabalha diretamente com os pixels brutos das imagens. Antes
disso, a biblioteca `face_recognition` transforma cada face em um vetor
numérico chamado embedding facial.

Durante o treino (alimentado por `dir_Treino`):

1. o código detecta a face em cada imagem;
2. extrai o embedding da face;
3. associa esse embedding ao nome da pessoa, que vem do nome da subpasta;
4. treina o KNN com esses pares.

Durante a predição (sobre imagens de `dir_Testes`):

1. o código detecta a face da imagem de teste;
2. extrai o embedding;
3. compara esse embedding com os embeddings do modelo treinado;
4. retorna a classe mais próxima, desde que a distância esteja abaixo do
   limiar definido.

## 6. Limiar de distância

O parâmetro `--distance-threshold` controla quando uma face deve ser aceita
como pertencente a uma pessoa conhecida.

- Valor menor: mais rigoroso, pode gerar mais `unknown`.
- Valor maior: mais permissivo, pode aumentar erros de identificação.

O valor padrão foi mantido como `0.6`, que é comum em exemplos da
biblioteca `face_recognition`.

## 7. Divisão treino/teste

A divisão é feita por classe, garantindo que cada pessoa tenha imagens
em ambos os conjuntos sempre que possível. Para classes com mais de uma
imagem, o código mantém pelo menos uma imagem em `dir_Treino` e pelo
menos uma em `dir_Testes`. Para classes com apenas uma imagem, ela
permanece em `dir_Treino`.

## 8. Saídas geradas

Após a execução, a pasta de saída terá uma estrutura semelhante a:

```text
output/face_knn/
  dataset_renomeado/
  dir_Treino/
  dir_Testes/
  models/
  predictions/
    run_1/
    run_2/
    run_3/
  logs/
    run_info.txt
    log_run_1.txt
    log_run_2.txt
    log_run_3.txt
    predictions_run_1.txt
    predictions_run_2.txt
    predictions_run_3.txt
    metrics_run_1_summary.txt
    metrics_run_1_per_class.tsv
    metrics_run_2_summary.txt
    metrics_run_2_per_class.tsv
    metrics_run_3_summary.txt
    metrics_run_3_per_class.tsv
    metrics_overall.tsv
```

### Configurações da execução (`run_info.txt`)

O arquivo `run_info.txt` registra os parâmetros usados no experimento,
os caminhos envolvidos, a hora de início, a hora de fim e a duração
total. O mesmo conteúdo é impresso no console no início e no final
da execução, garantindo que cada experimento seja autoexplicativo.

Inclui:

- caminho do `--dataset` original e do dataset em uso (que pode ser a
  cópia renomeada, em `dataset_renomeado/`);
- caminhos de `--workdir`, `dir_Treino` e `dir_Testes`;
- valores de `--porc`, `--inter`, `--n-neighbors`, `--distance-threshold`,
  `--seed` e do esquema de redimensionamento;
- timestamps de início e fim no formato `YYYY-MM-DD HH:MM:SS`;
- duração total formatada (segundos, minutos ou horas conforme o caso).

### Logs de estado

Os arquivos `log_run_*.txt` mostram onde cada imagem está em cada momento
do experimento. Colunas:

- `nomeO`: nome original da imagem;
- `diretorioO`: caminho original;
- `pastaO`: classe correta;
- `nomeM`: nome atual;
- `diretorioM`: caminho atual;
- `pastaM`: pasta atual;
- `em_teste`: indica se a imagem está em `dir_Testes` (teste);
- `ja_testada`: indica se a imagem já passou por `dir_Testes` em alguma
  execução.

### Logs de predição

Os arquivos `predictions_run_*.txt` mostram:

- `image_name`: imagem de teste;
- `expected_label`: classe esperada;
- `predicted_label`: classe predita;
- `correct`: se a predição coincidiu com a esperada;
- `face_location`: posição da face detectada.

### Logs de métricas

Após cada execução, três arquivos adicionais são gerados na pasta `logs/`:

- `metrics_run_N_summary.txt`: resumo legível da execução N, contendo
  totais de acertos, erros de identificação, rejeições e falhas de
  detecção, classes com pior recall e pares mais confundidos.
- `metrics_run_N_per_class.tsv`: métricas detalhadas por classe (uma
  linha por pessoa) com `tp`, `fn`, `fp`, precision, recall e F1.

Ao final de todas as execuções, é gerado também:

- `metrics_overall.tsv`: uma linha por execução mais uma linha `all`
  agregando todas, útil para acompanhar a estabilidade do classificador
  entre rodadas.

#### Como as métricas são definidas

No dataset Georgia Tech todas as imagens de teste contêm pessoas que
também aparecem no treino, então não há "verdadeiro negativo" no sentido
binário usual. As categorias adotadas são:

- **Acerto** (TP da classe): `predicted == expected`.
- **Erro de identificação**: o KNN respondeu uma pessoa errada. Conta
  como FP para a classe predita e FN para a classe esperada.
- **Rejeição (unknown)**: a distância ao vizinho mais próximo ficou acima
  do limiar. Conta como FN para a classe esperada.
- **Falha de detecção**: a biblioteca não encontrou face na imagem.
  Conta como erro, mas é uma categoria separada por não envolver o KNN.

Os cálculos ficam no módulo `metrics.py`. As principais funções:

- `compute_metrics(rows, run_number)`: recebe as linhas de predição e
  retorna um objeto `RunMetrics` com métricas globais, por classe e
  pares de confusão;
- `format_console_summary(metrics)`: produz um resumo enxuto para o
  terminal, mostrando totais globais, pior recall e top pares de
  confusão;
- `write_metrics_files(metrics, logs_dir)`: grava resumo + TSV por classe;
- `write_overall_summary(all_runs, logs_dir)`: grava o agregado.

## 9. Principais mudanças em relação ao código original

- Remoção do caminho fixo `D:/Documentos/...`.
- Parâmetros via linha de comando (`argparse`).
- Código separado em módulos menores e documentados.
- Substituição das classes manuais `Imagem` e `Pastas` por `dataclasses`.
- Troca de importações com `*` por importações explícitas.
- Normalização de extensões de imagem com `.lower()`.
- Inclusão de cabeçalhos nos logs.
- Substituição de `textsize` por `textbbox` para compatibilidade com
  versões novas do Pillow.
- Tratamento para o caso em que `k` é maior do que o número de faces
  válidas no treino.
- Mensagens de erro mais claras quando nenhuma face válida é encontrada.
- Ajuste da permuta para não tentar trocar mais imagens do que existem
  em `dir_Testes`.
- Uso de uma semente aleatória (`--seed`) para reprodutibilidade.
- Ajuste de nomenclatura: agora `dir_Treino` é a pasta que alimenta o
  treino do KNN, `dir_Testes` é a pasta que é predita, e `porc` é o
  percentual que permanece em treino — alinhando os nomes à convenção
  usual de aprendizado de máquina.

## 10. Exemplo completo de execução

```bash
python main.py \
  --dataset ./gt_db \
  --workdir ./output/face_knn \
  --porc 70 \
  --inter 3 \
  --n-neighbors 3 \
  --distance-threshold 0.6 \
  --resize-divisor 8
```

## 11. Observações finais

Este código ainda depende da qualidade do dataset. Para melhores resultados,
cada imagem deve conter uma única face visível, bem enquadrada e associada
corretamente à pasta da pessoa correspondente.

Se uma imagem tiver várias faces, ela será ignorada no treinamento para
evitar que o KNN aprenda um rótulo incorreto.
