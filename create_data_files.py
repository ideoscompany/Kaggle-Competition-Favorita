from Utils import load_data, save_unstack

# Carregar os dados
df_2017, promo_2017, items, stores = load_data()

# Salvar os dados em arquivos Feather
save_unstack(df_2017, promo_2017, '2017')