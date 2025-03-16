from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import gc

from Utils import *
import bisect

df_2017, promo_2017, items, stores = load_unstack("2017")

## Estrutura de Dados
# - df_2017: DataFrame principal com formato 'wide' onde:
#   - Índice multi-nível: (store_nbr, item_nbr)
#   - Colunas: datas (cada coluna é um dia)
#   - Valores: vendas transformadas com log1p
# - promo_2017: Mesmo formato que df_2017, mas com indicadores (0/1) para promoções
# - items: Informações sobre produtos (família, perecibilidade, etc.)
# - stores: Informações sobre lojas (tipo, cluster, etc.)

# Verificar o intervalo de datas disponíveis
print(f"Primeira data disponível: {min(df_2017.columns)}")
print(f"Última data disponível: {max(df_2017.columns)}")

# Remove os produtos que não tiveram vendas
promo_2017 = promo_2017[
    df_2017[pd.date_range(date(2015, 1, 1), date(2017, 8, 15))].max(axis=1) > 0
]
df_2017 = df_2017[
    df_2017[pd.date_range(date(2015, 1, 1), date(2017, 8, 15))].max(axis=1) > 0
]

# Converter promoções para inteiros para ganhar memória
promo_2017 = promo_2017.astype("int")

# Estrutura do DataFrame df_test:
# - Carregado do arquivo test.csv, contém os dados para os quais devemos fazer previsões
# - Índice multi-nível (hierárquico) com três componentes:
#   1. store_nbr: Identificador da loja (ex: 1, 2, 3...)
#   2. item_nbr: Identificador do produto (ex: 10203, 30457...)
#   3. date: Data da observação (convertida para timestamp)
# - Colunas relevantes:
#   * onpromotion: Indica se o item está em promoção (booleano True/False)
#   * unit_sales: Esta coluna não existe no teste, é o alvo da previsão
# - Cada linha representa uma combinação única de loja/produto/data
# - Os registros no df_test são usados para gerar as características (features)
#   através da função prepare_dataset() e depois fazer as previsões com
#   o modelo treinado para criar o arquivo de submissão final
df_test = pd.read_csv(
    "test.csv",
    usecols=[0, 1, 2, 3, 4],
    dtype={"onpromotion": bool},
    parse_dates=["date"],
).set_index(["store_nbr", "item_nbr", "date"])


# Filtragem dos itens nos dados de treinamento
# Esta etapa garante consistência entre os dados de treino e teste:
# 1. Extrai os identificadores de itens do conjunto de teste
# 2. Extrai os identificadores de itens do conjunto de treinamento
# 3. Encontra a interseção (itens comuns) entre ambos os conjuntos
# 4. Essa filtragem é crucial para:
#    - Evitar treinar o modelo com itens que não serão previstos
#    - Garantir que o modelo aprenda padrões apenas para itens relevantes
#    - Reduzir o tamanho do conjunto de dados e otimizar o treinamento
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df_2017.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

# Filtrar os dados de treinamento e promoção com base nos itens comuns
df_2017 = df_2017.loc[df_2017.index.get_level_values(1).isin(item_inter)]
promo_2017 = promo_2017.loc[promo_2017.index.get_level_values(1).isin(item_inter)]


# Função get_timespan: Extração de intervalos temporais dos dados
# Esta função permite extrair subconjuntos de dados para períodos específicos, sendo
# fundamental para a criação de características baseadas em padrões temporais:
# - df: DataFrame fonte (df_2017 ou promo_2017)
# - dt: Data de referência (ponto final do período)
# - minus: Quantos dias para trás a partir da data de referência
# - periods: Número de dias/períodos a serem incluídos
# - freq: Frequência dos períodos ('D' para dias, '7D' para semanas, etc.)
# A função lida com diferentes tipos de datas, convertendo-os para pandas.Timestamp
# quando necessário para garantir compatibilidade com os índices do DataFrame.
def get_timespan(df, dt, minus, periods, freq="D"):
    # Converter para pandas.Timestamp se for datetime.date
    if isinstance(dt, date):
        dt = pd.Timestamp(dt)
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


# Função prepare_dataset: Gera as características (features) para o modelo
# Esta função é o coração da engenharia de características (feature engineering) do projeto,
# criando um DataFrame rico em informações temporais e contextuais para cada combinação
# de loja/produto.
#
# Parâmetros:
# - t2017: Data de referência para a previsão (pivô temporal)
# - is_train: Se True, retorna também os valores alvo (y) para treinamento
# - one_hot: Se True, codifica variáveis categóricas com one-hot encoding
#
# A função gera múltiplas categorias de features:
# 1. Valores recentes: Vendas dos últimos 1-3 dias
# 2. Médias de períodos: Para janelas de 7, 14, 21, 30, 60, 90, 140, 365 dias
# 3. Métricas por período: Médias, medianas, máximos, contagem de zeros
# 4. Features de promoção: Quantidade de dias com promoção em diferentes períodos
# 5. Features condicionais: Vendas médias em períodos com/sem promoção
# 6. Agregações por item: Média e contagem de zeros por produto
# 7. Agregações por loja: Média e contagem de zeros por loja
# 8. Padrões por dia da semana: Médias específicas para cada dia da semana
# 9. Status promocional futuro: Indicadores de promoção para os próximos 16 dias
# 10. Informações de produto/loja: Família, perecibilidade, tipo de loja, etc.
#
# Para dados de treinamento, também retorna as vendas dos próximos 16 dias como alvo (y).
# O resultado é um conjunto de dados altamente informativo que captura padrões temporais
# de curto, médio e longo prazo, essenciais para previsões precisas de séries temporais.
def prepare_dataset(t2017, is_train=True, one_hot=False):
    # Converter para pandas.Timestamp se for datetime.date
    if isinstance(t2017, date):
        t2017 = pd.Timestamp(t2017)

    X = pd.DataFrame(
        {
            "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
            "day_2_2017": get_timespan(df_2017, t2017, 2, 1).values.ravel(),
            "day_3_2017": get_timespan(df_2017, t2017, 3, 1).values.ravel(),
            #         "day_4_2017": get_timespan(df_2017, t2017, 4, 1).values.ravel(),
            #         "day_5_2017": get_timespan(df_2017, t2017, 5, 1).values.ravel(),
            #         "day_6_2017": get_timespan(df_2017, t2017, 6, 1).values.ravel(),
            #         "day_7_2017": get_timespan(df_2017, t2017, 7, 1).values.ravel(),
            #         "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
            #         "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
            #         "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
            #         "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
            #         "median_30_2017": get_timespan(df_2017, t2017, 30, 30).median(axis=1).values,
            #         "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
            "promo_3_2017": get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
            "last_year_mean": get_timespan(df_2017, t2017, 365, 16).mean(axis=1).values,
            "last_year_count0": (get_timespan(df_2017, t2017, 365, 16) == 0)
            .sum(axis=1)
            .values,
            "last_year_promo": get_timespan(promo_2017, t2017, 365, 16)
            .sum(axis=1)
            .values,
        }
    )

    for i in [7, 14, 21, 30, 60, 90, 140, 365]:
        X["mean_{}_2017".format(i)] = (
            get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        )
        X["median_{}_2017".format(i)] = (
            get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        )
        X["max_{}_2017".format(i)] = (
            get_timespan(df_2017, t2017, i, i).max(axis=1).values
        )
        X["mean_{}_haspromo_2017".format(i)] = (
            get_timespan(df_2017, t2017, i, i)[
                get_timespan(promo_2017, t2017, i, i) == 1
            ]
            .mean(axis=1)
            .values
        )
        X["mean_{}_nopromo_2017".format(i)] = (
            get_timespan(df_2017, t2017, i, i)[
                get_timespan(promo_2017, t2017, i, i) == 0
            ]
            .mean(axis=1)
            .values
        )
        X["count0_{}_2017".format(i)] = (
            (get_timespan(df_2017, t2017, i, i) == 0).sum(axis=1).values
        )
        X["promo_{}_2017".format(i)] = (
            get_timespan(promo_2017, t2017, i, i).sum(axis=1).values
        )
        item_mean = (
            get_timespan(df_2017, t2017, i, i)
            .mean(axis=1)
            .groupby("item_nbr")
            .mean()
            .to_frame("item_mean")
        )
        X["item_{}_mean".format(i)] = df_2017.join(item_mean)["item_mean"].values
        item_count0 = (
            (get_timespan(df_2017, t2017, i, i) == 0)
            .sum(axis=1)
            .groupby("item_nbr")
            .mean()
            .to_frame("item_count0")
        )
        X["item_{}_count0_mean".format(i)] = df_2017.join(item_count0)[
            "item_count0"
        ].values
        store_mean = (
            get_timespan(df_2017, t2017, i, i)
            .mean(axis=1)
            .groupby("store_nbr")
            .mean()
            .to_frame("store_mean")
        )
        X["store_{}_mean".format(i)] = df_2017.join(store_mean)["store_mean"].values
        store_count0 = (
            (get_timespan(df_2017, t2017, i, i) == 0)
            .sum(axis=1)
            .groupby("store_nbr")
            .mean()
            .to_frame("store_count0")
        )
        X["store_{}_count0_mean".format(i)] = df_2017.join(store_count0)[
            "store_count0"
        ].values

    for i in range(7):
        X["mean_4_dow{}".format(i)] = (
            get_timespan(df_2017, t2017, 28 - i, 4, freq="7D").mean(axis=1).values
        )
        X["mean_10_dow{}".format(i)] = (
            get_timespan(df_2017, t2017, 70 - i, 10, freq="7D").mean(axis=1).values
        )
        X["count0_10_dow{}".format(i)] = (
            (get_timespan(df_2017, t2017, 70 - i, 10) == 0).sum(axis=1).values
        )
        X["promo_10_dow{}".format(i)] = (
            get_timespan(promo_2017, t2017, 70 - i, 10, freq="7D").sum(axis=1).values
        )
        item_mean = (
            get_timespan(df_2017, t2017, 70 - i, 10, freq="7D")
            .mean(axis=1)
            .groupby("item_nbr")
            .mean()
            .to_frame("item_mean")
        )
        X["item_mean_10_dow{}".format(i)] = df_2017.join(item_mean)["item_mean"].values
        X["mean_20_dow{}".format(i)] = (
            get_timespan(df_2017, t2017, 140 - i, 20, freq="7D").mean(axis=1).values
        )

    for i in range(16):
        # Converter para Timestamp antes de acessar
        t_day = pd.Timestamp(t2017 + timedelta(days=i))
        X["promo_{}".format(i)] = promo_2017[t_day].values

    if one_hot:
        family_dummy = pd.get_dummies(df_2017.join(items)["family"], prefix="family")
        X = pd.concat([X, family_dummy.reset_index(drop=True)], axis=1)
        store_dummy = pd.get_dummies(df_2017.reset_index().store_nbr, prefix="store")
        X = pd.concat([X, store_dummy.reset_index(drop=True)], axis=1)
    #         X['family_count'] = df_2017.join(items).groupby('family').count().iloc[:,0].values
    #         X['store_count'] = df_2017.reset_index().groupby('family').count().iloc[:,0].values
    else:
        df_items = df_2017.join(items)
        df_stores = df_2017.join(stores)
        X["family"] = df_items["family"].astype("category").cat.codes.values
        X["perish"] = df_items["perishable"].values
        X["item_class"] = df_items["class"].values
        X["store_nbr"] = df_2017.reset_index().store_nbr.values
        X["store_cluster"] = df_stores["cluster"].values
        X["store_type"] = df_stores["type"].astype("category").cat.codes.values
    #     X['item_nbr'] = df_2017.reset_index().item_nbr.values
    #     X['item_mean'] = df_2017.join(item_mean)['item_mean']
    #     X['store_mean'] = df_2017.join(store_mean)['store_mean']

    #     store_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('store_nbr').mean().to_frame('store_promo_90_mean')
    #     X['store_promo_90_mean'] = df_2017.join(store_promo_90_mean)['store_promo_90_mean'].values
    #     item_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('item_nbr').mean().to_frame('item_promo_90_mean')
    #     X['item_promo_90_mean'] = df_2017.join(item_promo_90_mean)['item_promo_90_mean'].values

    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y
    return X


# Preparação dos conjuntos de dados para treinamento, validação e teste
# Esta etapa define os três conjuntos de dados necessários para o fluxo de machine learning:
#
# 1. Conjunto de treinamento (X_train, y_train):
#    - Criado a partir de 14 semanas diferentes (n_range = 14)
#    - Começa da data 2017-07-05, retrocedendo de 7 em 7 dias
#    - Cada conjunto semanal contém features para todas as combinações loja/produto
#    - Os 14 conjuntos são concatenados para formar um conjunto de treinamento robusto
#    - Esta técnica (time-based cross-validation) captura melhor os padrões sazonais
#
# 2. Conjunto de validação (X_val, y_val):
#    - Gerado com data de referência 2017-07-26
#    - Usado para avaliar o modelo durante o treinamento e ajustar hiperparâmetros
#    - Permite verificar se o modelo generaliza bem para dados futuros
#
# 3. Conjunto de teste (X_test):
#    - Gerado com data de referência 2017-08-16
#    - is_train=False indica que não precisamos dos valores target (inexistentes)
#    - Usado para gerar as previsões finais para submissão
#
# Após concatenação, os objetos intermediários (X_l, y_l) são removidos para liberar memória.
print("Preparing dataset...")
X_l, y_l = [], []
t2017 = pd.Timestamp("2017-07-05")  # Em vez de date(2017, 7, 5)
n_range = 14
for i in range(n_range):
    print(i, end="..")
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(pd.Timestamp(t2017 - delta))
    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

# Criação dos conjuntos de validação e teste
# 
# Estas duas linhas configuram os dados para as etapas finais do pipeline de machine learning:
#
# 1. Conjunto de validação (X_val, y_val):
#    - Usa a data 2017-07-26 como referência (aproximadamente 3 semanas após o final do treino)
#    - Contém tanto features (X_val) quanto os valores alvo reais (y_val) para avaliação
#    - Será usado para monitorar o desempenho durante o treinamento e detectar overfitting
#
# 2. Conjunto de teste (X_test):
#    - Usa a data 2017-08-16 como referência (exatamente o período da competição)
#    - O parâmetro is_train=False indica que não precisamos dos valores target
#    - Contém apenas features (X_test) para predição das vendas finais
#
# Estas datas foram estrategicamente escolhidas para criar uma separação temporal 
# adequada entre treino, validação e teste, seguindo as melhores práticas para
# previsão de séries temporais.
X_val, y_val = prepare_dataset(pd.Timestamp(date(2017, 7, 26)))
X_test = prepare_dataset(pd.Timestamp(date(2017, 8, 16)), is_train=False)

# Configuração de hiperparâmetros do LightGBM
# Este dicionário define os parâmetros que controlam o comportamento do algoritmo:
#
# - num_leaves: 31 - Controla a complexidade de cada árvore (mais folhas = mais complexo)
# - objective: "regression" - Define o problema como regressão (prever valores contínuos)
# - min_data_in_leaf: 300 - Evita overfitting ao exigir muitos exemplos por nó folha
# - learning_rate: 0.05 - Taxa de aprendizado moderadamente conservadora para estabilidade
# - feature_fraction: 0.8 - Usa 80% das features em cada árvore (similar ao Random Forest)
# - bagging_fraction: 0.8 - Usa 80% dos dados em cada iteração (técnica de subamostragem)
# - bagging_freq: 2 - Aplica bagging a cada 2 iterações
# - metric: "l2" - Usa erro quadrático médio como métrica de avaliação
# - max_bin: 128 - Controla a granularidade da discretização de variáveis contínuas
# - num_threads: 8 - Utiliza 8 threads para processamento paralelo e aceleração
#
# Esta configuração equilibra precisão e generalização, sendo adequada para
# conjuntos de dados grandes como o da competição Favorita.
params = {
    "num_leaves": 31,
    "objective": "regression",
    "min_data_in_leaf": 300,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "metric": "l2",
    "max_bin": 128,
    "num_threads": 8,
}

print("Training and predicting models...")
MAX_ROUNDS = 700
val_pred = []
test_pred = []
best_rounds = []
cate_vars = ["family", "perish", "store_nbr", "store_cluster", "store_type"]
w = (X_val["perish"] * 0.25 + 1) / (X_val["perish"] * 0.25 + 1).mean()

for i in range(16):
    print("Step %d" % (i + 1))

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i], categorical_feature=cate_vars, weight=None
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val[:, i],
        reference=dtrain,
        weight=w,
        categorical_feature=cate_vars,
    )
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.callback.log_evaluation(period=100)],
    )

    print(
        "\n".join(
            ("%s: %.2f" % x)
            for x in sorted(
                zip(X_train.columns, bst.feature_importance("gain")),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
        )
    )
    best_rounds.append(bst.best_iteration or MAX_ROUNDS)

    val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(
        bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
    )
    gc.collect()

# Calcular e exibir o R² e RMSE para o conjunto de validação
val_pred = np.array(val_pred).T
r2 = r2_score(y_val, val_pred)
print(f"R² do conjunto de validação: {r2}")
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"RMSE do conjunto de validação: {rmse}")

cal_score(y_val, val_pred)

make_submission(df_2017, np.array(test_pred).T, "submission.csv")
