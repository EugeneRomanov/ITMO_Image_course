# 2024_processing_and_generating_images_course

# ФИО студента: Беседин Георгий Константинович

# Эксперименты с различными методами дистилляции знаний

## Описание проекта
В данном проекте исследуются различные методы дистилляции знаний (knowledge distillation) для передачи знаний от большой модели (Teacher) к меньшей модели (Student). Эксперименты проводятся на датасете CIFAR-10.

## Архитектуры моделей

### Teacher Network
```
- Входной слой: 3 канала (RGB)
- Конволюционные блоки:
  * Conv2d(3->64, kernel=3, padding=1) + ReLU
  * Conv2d(64->64, kernel=3, padding=1) + ReLU
  * MaxPool2d(2,2)
  * Conv2d(64->128, kernel=3, padding=1) + ReLU
  * Conv2d(128->128, kernel=3, padding=1) + ReLU
  * MaxPool2d(2,2)
- Полносвязные слои:
  * Linear(128*8*8->256) + ReLU
  * Linear(256->10)
```

### Student Network
```
- Входной слой: 3 канала (RGB)
- Конволюционные блоки:
  * Conv2d(3->32, kernel=3, padding=1) + ReLU
  * MaxPool2d(2,2)
  * Conv2d(32->64, kernel=3, padding=1) + ReLU
  * MaxPool2d(2,2)
- Полносвязные слои:
  * Linear(64*8*8->128) + ReLU
  * Linear(128->10)
```

## Параметры обучения

### Общие параметры
- Оптимизатор: SGD с momentum=0.9 и weight_decay=5e-4
- Learning rate: 0.01 с MultiStepLR (milestones=[15,25], gamma=0.1)
- Batch size: 128
- Количество эпох: 30
- Аугментация данных:
  * RandomCrop(32, padding=4)
  * RandomHorizontalFlip()
  * RandomRotation(15)
  * Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

### Параметры дистилляции

#### 1. Дистилляция логитов
- Температура (T): 4.0
- Вес cross-entropy loss (alpha): 0.1
- Вес KL-divergence: 1 - alpha = 0.9
- Функция потерь:
```python
loss = alpha * CE(student_logits, labels) + (1-alpha) * KL(student_logits/T, teacher_logits/T)
```

#### 2. Дистилляция скрытого состояния
- Метод сопоставления: усреднение по пространственным измерениям
- Размерность feature maps:
  * Teacher: 128 каналов -> используются первые 64
  * Student: 64 канала
- Вес cosine loss: 0.1
- Функция потерь:
```python
loss = CE(student_logits, labels) + 0.1 * cosine_loss(student_features, teacher_features)
```

#### 3. Дистилляция с регрессором
- Архитектура регрессора: Conv2d(128->64, kernel_size=1)
- Вес MSE loss: 0.2
- Функция потерь:
```python
loss = CE(student_logits, labels) + 0.2 * MSE(student_features, regressor(teacher_features))
```

#### 4. Комбинированная дистилляция
- Температура (T): 4.0
- Вес KL-divergence (alpha): 0.3
- Вес MSE (beta): 0.3
- Функция потерь:
```python
loss = CE(student_logits, labels) + alpha * KL(student_logits/T, teacher_logits/T) + beta * MSE(student_features, teacher_features)
```

## Результаты
| Модель/Метод | Точность на валидации | 
|--------------|----------------------|
| Teacher      | 84.26%               |
| Student (baseline) | 77.70%         |
| Logits distillation | 80.07%       | 
| Hidden state distillation | 78.10% | 
| Learnable regressor | 78.46%       | 
| Combined distillation | 80.43%     | 



## Выводы
1. Все методы дистилляции показали улучшение по сравнению с baseline
2. Наилучший результат показал комбинированный подход
3. Простая дистилляция логитов оказалась эффективнее, чем более сложные методы с feature mapping


