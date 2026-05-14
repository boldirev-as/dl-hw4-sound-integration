## Запуск на Kaggle

```bash
git clone https://github.com/boldirev-as/dl-hw4-sound-integration.git
cd dl-hw4-sound-integration
python3 -m pip install -r requirements.txt
python3 train.py
```

Датасет должен лежать тут: /kaggle/input/datasets/victorling/librispeech-clean

https://www.kaggle.com/datasets/victorling/librispeech-clean

## Инференс

```bash
python3 download_checkpoints.py
python3 infer.py
```

infer.py берет файл input.wav, прогоняет его и сохраняет результат в reconstructed.wav

## Demo

Ноутбук лежит в notebooks/demo.ipynb

## Результаты

https://www.comet.com/boldirev-as/dl-hw4-soundstream/view/new/panels

Лучшие метрики из Comet ML:

- STOI: 0.8515
- NISQA: 2.4803
- eval mel loss: 0.5805
