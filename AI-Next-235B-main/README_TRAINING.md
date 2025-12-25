# Обучение мировой нейросети на датасете The Pile

Добро пожаловать в проект по обучению мировой нейросети с 235 миллиардами параметров на датасете The Pile.

## Структура проекта

- `nn_architecture.py` - Архитектура нейросети с 101 слоем, включая специальный слой управления контекстом по матрице Эйзенхауэра
- `context_management_layer.py` - Специализированный слой для анализа контекста
- `gui_context_manager.py` - Графический интерфейс для визуализации контекстного анализа
- `train_pile.py` - Основной скрипт для обучения на датасете The Pile
- `setup_train.sh` - Скрипт установки зависимостей
- `TRAINING_GUIDE.md` - Подробное руководство по обучению
- `ds_config.json` - Конфигурационный файл для DeepSpeed (пример)

## Требования к оборудованию

Для успешного обучения модели с 235 миллиардами параметров требуются:

- 8 или более GPU NVIDIA A100 (80GB) или H100
- Не менее 1 ТБ оперативной памяти (рекомендуется 2 ТБ)
- Высокоскоростная сеть между узлами (InfiniBand)
- Не менее 5 ТБ дискового пространства для чекпоинтов и логов
- 100 Гбит/с сеть для распределенного обучения

## Установка зависимостей

```bash
bash setup_train.sh
```

## Подготовка датасета The Pile

1. Скачайте датасет The Pile с официального зеркала: https://the-eye.eu/public/AI/pile/
2. Распакуйте .jsonl файлы в директорию `/workspace/pile_data/`

## Запуск обучения

### Локальный запуск (только для тестирования):

```bash
python train_pile.py
```

### Распределенное обучение с DeepSpeed:

```bash
# Создание конфигурационного файла DeepSpeed
cat << EOF > ds_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e9,
        "stage3_prefetch_bucket_size": 1e9,
        "stage3_param_persistence_threshold": 1e9
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
EOF

# Запуск обучения
deepspeed --num_gpus=8 train_pile.py --deepspeed ds_config.json
```

## Архитектура модели

Наша нейросеть включает в себя:

- 101 слой (2 Embedding/Positional + 96 Transformer + 3 Final)
- 235 миллиардов параметров
- 12 идентичных суперблоков с 8 подслоами каждый
- 256 экспертов на слой (MoE)
- 2 специальных токена для классификации по интонации и научной области
- 42 активных эксперта на токен
- Специализированный слой управления контекстом по матрице Эйзенхауэра
- Gated DeltaNet, Grouped Query Attention, RMSNorm и другие передовые технологии

## Графический интерфейс

Приложение включает визуализацию работы слоя управления контекстом:

```bash
python gui_context_manager.py
```

## Мониторинг обучения

Для отслеживания процесса обучения рекомендуется использовать Weights & Biases:

```python
import wandb
wandb.init(project="world-class-neural-network", config=training_config)
```

## Проверка конфигурации

Перед запуском обучения проверьте конфигурацию системы:

```bash
python check_config_simple.py
```

## Результаты и чекпоинты

Чекпоинты сохраняются в директорию `./checkpoints/`:
- После каждой эпохи
- Лучшая модель как `best_model.pth`
- Регулярные сохранения для восстановления после сбоев

## Ожидаемая производительность

Модель разработана для достижения уровня производительности, сравнимого с GPT-4o, с возможностью понимания сложных контекстов благодаря инновационному слою управления контекстом по матрице Эйзенхауэра.

## Лицензия

Проект лицензирован под лицензией MIT.