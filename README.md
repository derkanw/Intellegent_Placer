# Intellegent Placer
## Постановка задачи
### Описание задачи
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольника определить, помещаются ли каким-либо способом данные предметы в
этот многоугольник. При этом все предметы и поверхность заранее известны.

### Входные и выходные данные
**Входные данные:** фотография предметов и многоугольника на светлой горизонтальной плоскости

**Выходные данные:** строка формата "[ответ] [имя исходной фотографии]" в файле answers.txt в корне проекта. Ответ принимает вид "True" или "False" в зависимости от того, входят ли предметы в заданный многоугольник или нет соответсвенно.

### Общие требования
- Задается набор входных данных
- Задается набор отдельных рассматриваемых предметов
- Фотографируется отдельно рассматриваемая поверхность

#### Требования к фотографиям
- Без дополнительного сжатия, цветовой коррекции
- Формат фотографии png\jpg
- Имя фотографии имеет вид [название предмета] или [название примера]_[ожидаемый результат]
- Фотографии должны быть сделаны на одно и то же устройство
- Освещение фотографии должно быть одинаковым, равномерным (без засветов и темных областей)
- Фотография входных данных содержит только заранее известные предметы и многоугольник
- На фотографии исходных предметов каждый предмет фотографируется в едином экземпляре
- Съемка производится горизонтально с допустимым отклонением (возможна ошибка до 10°)
- Высота съемки едина для всех фотографий (допускается отклонение до 3 см)
- Все фотографии одной и тоже ориентации: книжная (вертикальная) или альбомная (горизонтальная)

#### Требования к поверхности
- Ровная однородная горизонтальная поверхность
- Единая для всех фотографий

#### Требования к предметам
- Каждый предмет распологается на чистом листе бумаги
- Предмет или их набор находятся в центре листа и не выходят за его пределы
- Края данного листа видны на фотографии и не сливаются с поверхностью
- Предметы в наборе не перекрывают друг друга и не имеют общих границ
- Края предметов четко отличимы друг от друга, поверхности и листа бумаги (не менее 5 мм расстояния между предметами)
- Тень от предмета должна быть площадью не более 10% от площади предмета

#### Требование к многоугольнику
- Задается на отдельном листе бумаги черным маркером
- Данный лист фотографируется вместе с набором предметов для задачи
- Набор предметов и рассматриваемый для них многоугольник не пересекаются
- Толщина линии маркера до 5 мм
- Нарисован в центре листа бумаги
- Многоугольник должен быть выпуклым
- Число вершин многоугольника должно быть не более 10

## Набор данных
[Датасет](https://drive.google.com/drive/folders/1S9s03F0Fk_Z-EFmSU9u3fNcNZpPfgtCL?usp=sharing)

## Алгоритм
### План алгоритма
#### Обработка изображений исходных предметов
1. Провести бинаризацию изображения
2. Найти область отображения предмета с помощью threshold_otsu
3. Сгладить маску посредством морфологического закрытия
4. Избавиться от шумов морфологическим раскрытием
5. Получить свойства области отображения предмета
6. Построить и применить маску к изображению
7. Обрезать изображение по области отображения предмета
8. Найти особые точки и их дескрипторы

#### Определение предметов и многоугольника на входном изображении
1. Выделить объекты в виде разности входного изображения и изображения поверхности
2. Найти границы объектов с помощью детектора canny
3. Провести сглаживание посредством морфологического закрытия
4. Заполнить области внутри найденных контуров
5. Исключить шум с помощью морфологического раскрытия
7. Получить свойства объектов, опираясь на центр масс изображения
8. Выделить многоугольник как объект с наименьшей ординатой
9. Обрезать изображение по области отображения для остальных объектов
10. Найти особые точки и их дескрипторы
11. Распознать предметы на входном изображении с помощью максимального соответствия по доле особых точек данных предметов с исходными

#### Размещение предметов в многоугольнике
1. Найти сумму площадей рассматриваемых предметов
2. Сравнить данную сумму с площадью многоугольника

### Результаты работы
Алгоритм настроен на работу с одним вариантом входных данных (Example.jpg), 
правильно обрабатывает набор исходных предметов и распознает предмет на входном изображении с совпадением по особым точкам с долей, равной 0.72.

Исходя из особенностей входного изображения, тривиальный алгоритм размещения объектов в многоугольнике дает верный ответ.

### TODO
- Реализовать укладку объектов в многоугольник
- Обновить и обработать полный набор входных данных
- Обработать особые случаи и возможные ошибки
- Оформить заданный в ТЗ вид вывода

### Улучшения
- Автоматический подбор параметров
- Анализ threshold и выбор наиболее подходящего
