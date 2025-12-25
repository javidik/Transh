# Обучение модели на датасете The Pile

## Подготовка к обучению

Для запуска процесса обучения вашей нейросети на датасете The Pile выполните следующие шаги:

1. Установите зависимости:
```bash
bash setup_train.sh
```

2. Загрузите датасет The Pile:
   - Перейдите на сайт: https://the-eye.eu/public/AI/pile/
   - Скачайте нужные .jsonl файлы
   - Поместите их в директорию `/workspace/pile_data/`

## Настройка параметров обучения

В скрипте `train_pile.py` можно настроить следующие параметры:

- `batch_size`: Размер батча (уменьшите при нехватке памяти)
- `accumulation_steps`: Количество шагов для аккумуляции градиентов
- `learning_rate`: Скорость обучения
- `epochs`: Количество эпох обучения
- `max_length`: Максимальная длина последовательности
- `warmup_ratio`: Процент шагов для прогрева обучения

## Запуск обучения

Для запуска обучения выполните:
```bash
python train_pile.py
```

## Использование DeepSpeed для распределенного обучения

Для обучения модели размером 235 миллиардов параметров рекомендуется использовать DeepSpeed. Создайте конфигурационный файл DeepSpeed:

```bash
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
```

Затем запустите обучение с использованием DeepSpeed:
```bash
deepspeed --num_gpus=8 train_pile.py --deepspeed ds_config.json
```

## Проверка прогресса обучения

В процессе обучения отслеживайте:
- Значение функции потерь (loss)
- Уровень точности на валидационной выборке
- Использование памяти GPU
- Значения метрик сходимости

## Сохранение контрольных точек

Модель автоматически сохраняет контрольные точки после каждой эпохи в директорию `./checkpoints/`. Лучшая модель сохраняется как `best_model.pth`.

## Требования к оборудованию

Для обучения модели с 235 миллиардами параметров рекомендуются:
- 8 или более GPU NVIDIA A100 (80GB)
- Не менее 1 ТБ оперативной памяти
- Высокоскоростная сеть между узлами (InfiniBand)
- Достаточное дисковое пространство для хранения чекпоинтов

## Мониторинг и отладка

Для отслеживания экспериментов используйте Weights & Biases:
```python
import wandb
wandb.init(project="world-class-neural-network")
```

## Возможные проблемы и решения

1. **Нехватка видеопамяти**: Уменьшите batch_size или используйте градиентный аккумулятор
2. **Высокое использование CPU**: Увеличьте количество worker-ов в DataLoader
3. **Переобучение**: Увеличьте значение dropout или добавьте регуляризацию
4. **Низкая сходимость**: Проверьте гиперпараметры обучения и начальную инициализацию весов