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

Precisamos então avaliar o classificador, e nesse momento poderemos utilizar a API fornecida pelo professor para calcular a acurácia do modelo. Entretanto ela possui um limite de 13 minutos entre cada submissão, então buscamos criar uma solução mais performática para isso. A solução encontrada foi então utilizar a técnica **Hold out**, dividindo as 196 linhas resultantes da base de dados em base de treinamento e base de teste, a um fator de `0.9` de base de teste e, consequentemente, `0.1` de base de treinamento. Com isso teremos então 176 linhas de dados para treinamento e 19 de testes.

Agora como medida de desempenho temos a acurácia como principal medida do modelo, sendo essa a utilizada pelo professor em sua API. Entretanto, como estaremos utilizando também uma base de treino e teste local, calculamos também a matriz de confusão, com sua acurácia, precision, recall e f-score. Com essas informações a mais, poderemos avaliar também como está ocorrendo a distribuição de acertos e erros da base treinada.

Nesse momento, temos como acurácia, pela API, o valor `0.6275`, enquanto pela base de teste o valor `0.6667`. Além disso, na base de teste temos `precisão = 0.5312`, `recall = 0.2787` e `f-score = 0.3656`.

- Seed 42:

1. StandardScaler em Pregnancies:
   - Acurácia (API) = ? ; Acurária (Teste) = 0.6667 ; Precisão (Teste) = 0.5312 ; Recall (Teste) = 0.2787 ; F-Score (Teste) = 0.3656
2. StandardScaler em SkinThickness
   - Acurácia (API) = 0.6122 ; Acurária (Teste) = 0.6723 ; Precisão (Teste) = 0.5455 ; Recall (Teste) = 0.2951 ; F-Score (Teste) = 0.3830
3. StandardScaler em Insulin
   - Acurácia (API) = ? ; Acurária (Teste) = 0.7006 ; Precisão (Teste) = 0.6176 ; Recall (Teste) = 0.3443 ; F-Score (Teste) = 0.4421
4. StandardScaler em BMI
   - Acurácia (API) = ? ; Acurária (Teste) = 0.7175 ; Precisão (Teste) = 0.6774 ; Recall (Teste) = 0.3443 ; F-Score (Teste) = 0.4565
5. StandardScaler em Age
   - Acurácia (API) = 0.6173 ; Acurária (Teste) = 0.7627 ; Precisão (Teste) = 0.7436 ; Recall (Teste) = 0.4754 ; F-Score (Teste) = 0.5800
6. Translação de Glucose em quartil 0.5
   - Acurácia (API) = 0.6275 ; Acurária (Teste) = 0.7288 ; Precisão (Teste) = 0.6667 ; Recall (Teste) = 0.4262 ; F-Score (Teste) = 0.5200
7. Translação de BloodPressure em quartil 0.5
   - Acurácia (API) = 0.6275 ; Acurária (Teste) = 0.7514 ; Precisão (Teste) = 0.6977 ; Recall (Teste) = 0.4918 ; F-Score (Teste) = 0.5769
8. Dilatação de Pregnancies em 2
   - Acurácia (API) = ? ; Acurária (Teste) = 0.7401 ; Precisão (Teste) = 0.6714 ; Recall (Teste) = 0.4754 ; F-Score (Teste) = 0.5577
9. Dilatação de Glucose em 2
   - Acurácia (API) = ? ; Acurária (Teste) = 0.7458 ; Precisão (Teste) = 0.6600 ; Recall (Teste) = 0.5410 ; F-Score (Teste) = 0.5946
10. Dilatação de BloodPressure em 2
    - Acurácia (API) = ? ; Acurária (Teste) = 0.7514 ; Precisão (Teste) = 0.6977 ; Recall (Teste) = 0.4918 ; F-Score (Teste) = 0.5769
11. Dilatação de DiabetesPedigreeFunction em 25
    - Acurácia (API) = 0.6173 ; Acurária (Teste) = 0.7627 ; Precisão (Teste) = 0.7209 ; Recall (Teste) = 0.5082 ; F-Score (Teste) = 0.5962

- Seed 10:

1. Remoção de Dilatação de Pregnancies em 2
   - Acurácia (API) = ? ; Acurária (Teste) = 0.6271 ; Precisão (Teste) = 0.4000 ; Recall (Teste) = 0.1639 ; F-Score (Teste) = 0.2326
2. Remoção de Translação de Glucose em quartil 0.5
   - Acurácia (API) = ? ; Acurária (Teste) = 0.6723 ; Precisão (Teste) = 0.5652 ; Recall (Teste) = 0.2131 ; F-Score (Teste) = 0.3095
3. Atualização de Dilatação de Glucose em 2 para 0.5
   - Acurácia (API) = 0.6377 ; Acurária (Teste) = 0.6780 ; Precisão (Teste) = 0.7500 ; Recall (Teste) = 0.0984 ; F-Score (Teste) = 0.1739
