# Aprendizagem

O processo de descoberta de conhecimento é composto por 6 fases:

1. Seleção dos dados
2. Limpeza dos dados
3. Enriquecimento dos dados
4. Transformação dos dados
5. Mineração dos dados
6. Apresentação e análise dos resultados

Para esse projeto estaremos utilizando como classificador o algoritmo de aprendizagem supervisionada **KNN** com `n = 3`.

Primeiro, vamos compreender o que cada coluna da nossa base de dados representa:

- **Pregnancies**: número de vezes grávida
- **Glucose**: concentração plasmática de glicose a 2 horas em um teste oral de tolerância à glicose
- **BloodPressure**: pressão arterial diastólica (mm Hg)
- **SkinThickness**: espessura da dobra da pele do tríceps (mm)
- **Insulin**: insulina sérica de 2 horas (mu U/ml)
- **BMI**: índice de massa corporal (peso em kg / (altura em m) ^ 2)
- **DiabetesPedigreeFunction**: função de pedigree do diabetes
- **Age**: idade (anos)
- **Outcome**: variável de classe (0 ou 1) para diabetes

No nosso problema teremos então que aplicar um classificador que utilizará como atributo dependente a coluna **Outcome** e atributos independentes todas as outras colunas.

Essa base possui 572 linhas de dados, onde vamos limpar os dados removendo as linhas que possuem ao menos uma coluna com o valor não preenchido, mantendo então 196 linhas. Com isso já podemos iniciar uma outra análise dos dados, agora olhando para seu conteúdo e correlação entre eles. Nesse momento ainda não enriquecemos os dados ou os transformamos, apenas pretendemos deixar a base pronta para uma aplicação do algoritmo **KNN** sem erros.

Precisamos então avaliar o classificador, e nesse momento poderemos utilizar a API fornecida pelo professor para calcular a acurácia do modelo. Entretanto ela possui um limite de 13 minutos entre cada submissão, então buscamos criar uma solução mais performática para isso. A solução encontrada foi então utilizar a técnica **Hold out**, dividindo as 196 linhas resultantes da base de dados em base de treinamento e base de teste, a um fator de `0.3` de base de teste e, consequentemente, `0.7` de base de treinamento. Com isso teremos então 137 linhas de dados para treinamento e 59 de testes.

Agora como medida de desempenho temos a acurácia como principal medida do modelo, sendo essa a utilizada pelo professor em sua API. Entretanto, como estaremos utilizando também uma base de treino e teste local, calculamos também a matriz de confusão, com sua acurácia, precision, recall e f-score. Com essas informações a mais, poderemos avaliar também como está ocorrendo a distribuição de acertos e erros da base treinada.

Nesse momento, temos como acurácia, pela API, o valor `0.5612`, enquanto pela base de teste o valor `0.5816`. Além disso, na base de teste temos `precisão = 0.4118`, `recall = 0.4667` e `f-score = 0.4375`.

Observamos na base de dados, principalmente, a correlação entre as colunas. As correlações que mais se destacaram foram:

- **Glucose e Insulin:** 0.601148
- **SkinThickness e BMI:** 0.635335

Além disso, observando o gráfico boxplot da base, vimos que os dados estão bastante espaçados entre si. Então, com essas duas informações, buscamos primeiro normalizar os dados e, novamente, observar a correlação entre as colunas. A técnica escolhida para a primeira normalização dos dados foi a Regularização L2. Com ela, buscamos diminuir a espacialidade entre os dados e mante-los entre valores próximos de 0 a 1. Além disso, essa técnica de normalização ajusta levemente os dados para diminuir a possibilidade de overfitting, como também modifica mais fortemente colunas com valores maiores, diminuindo a magnitude dos valores mas sem alterar a proporção entre eles. Com ele organizaremos os dados em uma escala comum e ainda diminuiremos a possibilidade de overfitting, especialmente em uma base de dados que no momento é pequena e possui colunas com correlação baixa entre si.

Aplicada a normalização l2, Temos como acurária pela API o valor `0.5510`, enquanto pela base de teste o valor `0.6949`, Além disso, na base de teste temos `precisão = recall = f-score = 0.4`. Podemos nesse momento observar, pela base de testes, uma diminuição de verdadeiros positivos e, na mesma proporção, um aumento de verdadeiros negativos. Além disso, a distribuição de falsos negativos e falsos positivos flutuou para que agora estejam iguais, mantendo a base de treinamento muito mais balanceada. Com isso, mesmo com uma diminuição pequena na acurácia pela API, optamos por manter essa técnica.

Agora, voltamos a observar a matrix de correlação entre as colunas. Com isso, observamos mais uma vez o seguinte comportamento:

- **Glucose e Insulin:** -0.804723
- **BloodPressure e Insulin:** -0.835506
- **SkinThickness e BMI:** 0.774038
- **BMI e Insulin:** -0.685238
- **Age e Pregnancies:** 0.671700
- **Age e Insulin:** -0.605631

Mesmo com a correlação de Glucore e Insulin sendo semelhante a BloodPressure e Insulin, a correlação entre Glucose e BloodPressure é de `0.511404`, o que mostra que o ângulo está bem próximo de 45 graus entre os vetores de correlação dessas duas colunas, além de ter valores bem espaçados entre si. Também vemos uma correlação natural mais evidente entre a idade e número de gestações. Entretanto, também temos que a correção entre Pregnancies e Insulin é de `-0.217418`, sendo essa muito baixa para justificar alguma modificação. Acabamos por observar que Glucose possui muita correlação com Insulin, mas essa não possui, na mesma proporção, correlações com outras colunas tal qual Insulin. Sendo assim, mesmo elas tendo correlações próximas, não parece uma boa ideia trabalhar no momento com alterações entre elas.

Voltamos então para as colunas SkinThickness e BMI, que possuem alta correlação entre si e suas correlações com outras colunas continuam na mesma proporção. Por exemplo, BMI e Insulin possuindo correlação `-0.685238` enquanto SkinTickness e Insulin `-0.528451`. Escolhemos por enfim observar a matriz de correlação entre essas duas colunas, concluindo que sua distribuição é bastante organizada para valores mais baixos, tendo apenas um pouco mais de desordem quando os valores de ambas estão bem mais altos. Percebemos também que os dados são bastante representativos para valores entre _(0,0)_ e _(0.3,0.3)_, sendo pouco representativos para valores fora desse intervalo, especialmente entre _(0.3,0.3)_ e _(0.4,0.4)_ .Dentre BMI e SkinTickness, podemos então reduzir a dimensionalidade do modelo utilizando a técnica PCA que, para colunas com grau de similaridade inversamente proporcionais, captura boa parte da variância em um único componente, resultando em uma nova coluna.

Criada essa nova coluna, removemos as duas anteriores e obtivemos acurácia `0.5816` pela API, ainda `0.6949` pela base de testes, mas agora com `precisão = 0.3846`, `recall = 0.3333` e `f-score = 0.3571`. Observamos um modelo agora predizendo mais valores como negativos sem necessariamente melhorar significativamente a acurácia, diminuindo apenas um pouco a precisão. Esperamos com isso ter uma base mais enxuta para melhorar as possibilidades de aumento de acurácia analisando melhor a base resultante.

Vamos com isso repetir o mesmo experimento de redução de dimensionalidade mas agora entre as colunas Glucose e Insulin. Feito isso, teremos também como resultados uma acurácia de `0.5408` pela API, `0.7288` pela base de testes, `precisão = 0.4762`, `recall = 0.6667` e `f-score = 0.5556`. Conseguimos dessa vez predizer melhor verdadeiros positivos, um pouco pior os verdadeiros negativos mas, por fim, aumentar a precisão e acurácia. Observamos uma flutuação significativa do modelo agora prevendo mais valores como positivos. Como houve uma diminuição, novamente, na acurácia da API, acreditamos que há valores não cobertos pela base de testes, o que pode ser ocasionado pela diminuição na expressividade dos dados, causada pela primeira escolha de limpar os dados mantendo apenas linhas com informações preenchidas.

Decidimos por fim voltar na limpeza dos dados. Observamos dentre os dados faltantes que:

- 374 linhas não possuem a coluna Insulin preenchida
- 227 linhas não possuem a coluna SkinThickness preenchida
- 35 linhas não possuem a coluna BloodPressure preenchida
- 11 linhas não possuem a coluna BMI preenchida
- 5 linhas não possuem a coluna Glucose preenchida

Lembrando que Insulin e Glucose possuem, sem tratamento, grau de correlação de `0.601148`, podemos tentar inferir os valores de faltantes de Insulin a partir disso. Decidimos por aplicar a técnica de regressão linear sobre os valores de Glucose para inferir Insulin quando não houver valor para Insulin. Agora temos então em nossa base de dados 235 dados de treino e 101 dados de teste. Aplicando a mesma técnica anterior, agora obtivemos acurácia de `0.5816` pela API, `0.6436` pela base de teste, `precisão = 0.4800`, `recall = 0.3429` e `f-score = 0.4000`. Podemos observar então que agora nossa base de treino obteve acurácia melhor contra a base da API e uma perda significativa de acurácia contra a base de testes, Entretanto, como a precisão aumentou, podemos inferir que a base encontra-se mais balanceada.

Repetindo o processo para a coluna SkinTickness, vamos aplicar também a técnica de regressão linear sobre a coluna BMI. Aumentamos novamente nossa base de dados para 369 dados de treino e 159 dados de testes. Com a mesma técnica anterior de preprocessamento, obtivemos acurácia de `0.5714` pela API, `0.6855` pela base de teste, `precisão = 0.4828`, `recall = 0.5833` e `f-score = 0.5283`. Novamente, nossa base parece representar melhor a realidade, agora com melhorias no recall e f-score.

Por fim, temos agora a seguinte quantidade de colunas vazias na base de dados: `Glucose = 5`, `BloodPressure = 35`, `SkinThickness = 9`, `Insulin = 4` e `BMI = 11`. Feito o enriquecimento dos dados, vamos voltar a trabalhar na transformação deles.

Analisando novamente a distribuição dos dados após o preprocessamento, vemos que agora a coluna `glucose_insulin` está com variação entre -1 e 1. Testamos então aplicar a normalização Min-Max sobre essa coluna para devolve-la a valores entre 0 e 1. Obtivemos agora acurácia de `0.5765` contra a API, `0.6918` contra a base de teste, `precisão = 0.4906`, `recall = 0.5417` e `f-score = 0.5149`. Observamos uma melhora na predição de valores negativos ao mesmo passo que uma piora na predição de valores positivos.

Agora vamos aplicar a mesma normalização Min-Max sobre a coluna `skinthickness_bmi`. Obtivemos como resultados contra a API a acurácia de `0.5765`, enquanto `0.6981` contra a base de testes, `precisão = 0.5000`, `recall = 0.6042` e `f-score = 0.5472`. Melhoramos com isso, principalmente, a precisão do modelo, sem alterar a acurácia.

Percebemos que a coluna `DiabetesPedigreeFunction` estava com dados muito espaçados e com variação entre 0 e 0.04, média 0.0046 e primeiro e terceiro quartis entre 0.002 e 0.006. Isso se dava por vários outliers na base, incluindo um único que deslocava o valor máximo sozinho. Por isso, decidimos por combinar duas técnicas: remoção de outliers e min-max. Conseguimos com isso `0.5714` de acurácia pela API, `0.7020` pela base de teste, juntamente de `precisão = 0.5686`, `recall = 0.5577` e `f-score = 0.5631`. Assim, deslocamos as predições para acertar mais os verdadeiros positivos em troca de errar um pouco mais os verdadeiros negativos.
