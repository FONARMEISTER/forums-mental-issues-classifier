# Данные

`data.csv` - исходный корпус

`data_filtered.csv` - отфильтрованный deepseek-v4-flash корпус

`augmented_data.csv` - сгенерированный deepseek-v4-flash синтетический корпус

# Код

`analysis.opynb` - аналитика по корпусу

`bert_filtered_dataset.py` - fine-tuning BERT-а на корпусе `data_filtered.csv`

`tfidf_raw_dataset.py` - Жесткая фильтрация с маркерами, TFIDF эмбеддинги и SVM-классификатор на корпусе `data.csv`

`tfidf_raw_dataset.ipynb` - Копия `tfidf_raw_dataset.py`, в более читаемом формате блокнота


# Логи

`bert.log` - логи запуска `bert_filtered_dataset.py`

`tfidf.log` - логи запуска `tfidf_raw_dataset.py`
