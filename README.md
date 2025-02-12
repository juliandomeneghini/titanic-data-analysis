# Titanic - Análise e Previsão de Sobrevivência

Este repositório contém um projeto de análise de dados e previsão de sobrevivência baseado no famoso *Titanic dataset* do Kaggle. O objetivo é explorar os dados, visualizá-los e, por fim, treinar modelos de Machine Learning para prever a sobrevivência dos passageiros com base em características como idade, sexo, classe do bilhete, entre outras.

## Passos do Projeto

1. **Carregamento e Exploração do Dataset**:
   O dataset foi carregado e explorado para verificar informações sobre as variáveis e identificar possíveis valores ausentes. Realizamos uma análise descritiva e visualizamos a distribuição dos dados.

2. **Análises Visuais**:
   Através do uso das bibliotecas **Matplotlib** e **Seaborn**, foram realizadas visualizações dos dados, incluindo gráficos de:
   - Sobrevivência por sexo
   - Distribuição da idade
   - Sobrevivência por faixa etária
   - Sobrevivência por classe de passagem
   - Sobrevivência por tamanho da família
   - Distribuição da tarifa e sobrevivência por faixa de tarifa.

3. **Pré-processamento de Dados**:
   - Preenchimento de valores ausentes.
   - Conversão de variáveis categóricas em variáveis numéricas.
   - Criação de novas variáveis baseadas em análises de dados (como a coluna `FamilySize` e `FareGroup`).

4. **Divisão dos Dados**:
   - Os dados foram divididos em variáveis independentes (X) e a variável dependente (y).
   - O dataset foi dividido em conjunto de treino e teste (80% para treino e 20% para teste).

5. **Modelos de Machine Learning**:
   - **Regressão Logística**: Foi treinado um modelo de Regressão Logística para prever a sobrevivência dos passageiros.
   - **Random Forest Classifier**: Utilizamos o modelo Random Forest para melhorar a precisão da previsão, ajustando seus hiperparâmetros para encontrar o melhor desempenho.

6. **Avaliação do Modelo**:
   - A acurácia foi calculada para avaliar a precisão do modelo.
   - A matriz de confusão foi gerada para visualizar os resultados das previsões.

## Resultados

- A acurácia do modelo Random Forest após ajuste de hiperparâmetros foi de **82.68%**.
- A matriz de confusão foi gerada para avaliar como o modelo classificou as previsões de sobrevivência, com as classes **"Sobreviveu"** e **"Não Sobreviveu"**.

## Instalação

Este projeto requer as seguintes bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Como Usar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/random-forest-classifier.git
   ```

2. Faça o download do arquivo `titanic.csv` do Kaggle e coloque-o na pasta do projeto.

3. Execute o script para realizar a análise e treinar os modelos de Machine Learning.

4. Os resultados, como a acurácia e a matriz de confusão, serão exibidos ao final da execução.

## Exemplos de Resultados

Abaixo está o código utilizado para treinar o modelo e gerar as visualizações:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Inicializando o modelo Random Forest com os melhores parâmetros encontrados
best_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)

# Treinando o modelo
best_rf_model.fit(X_train, y_train)

# Fazendo previsões com o modelo treinado
rf_y_pred = best_rf_model.predict(X_test)

# Avaliando a acurácia
print(f'Acurácia (Random Forest - Melhor Hiperparâmetro): {accuracy_score(y_test, rf_y_pred)}')

# Exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, rf_y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Sobreviveu', 'Sobreviveu'], yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.xlabel('Previsões')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
```

## Contribuições

Sinta-se à vontade para contribuir com melhorias ou sugestões para este repositório. Caso tenha alguma dúvida ou queira discutir mais sobre o código, abra uma **issue**.

---

**Autor**: Seu Nome  
**Data**: Fevereiro 2025  
