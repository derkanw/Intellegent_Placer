# Intellegent Placer
## Постановка задачи
### Описание задачи
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольника определить, помещаются ли каким-либо способом данные предметы в
этот многоугольник. При этом все предметы и поверхность заранее известны.

### Входные и выходные данные
**Входные данные:** фотография предметов и многоугольника на светлой горизонтальной плоскости
**Выходные данные:** строка формата "[ответ] [имя исходной фотографии]" в файле answers.txt в корне проекта. Ответ принимает вид "True" или "False" в зависимости от того, входят ли предметы в заданный многоугольник или нет соответсвенно.

### Общие требования
- Задается набор входных данный
- Задается набор отдельных расматриваемых предметов
- Фотографируется отдельно рассматриваемая поверхность

#### Требования к фотографиям
- Без сжатия, цветовой коррекции
- Формат фотографии png\jpg
- Имя фотографии имеет вид [название предмета] или [название примера]_[ожидаемый результат]
- Фотографии должны быть сделаны на одно и то же устройство
- Освещение фотографии должно быть одинаковым, равномерным (без засветов и темных областей)
- Фотография входных данных содержит только заранее известные предметы и многоугольник
- На фотографии исходных предметов каждый предмет фотографируется в едином экземпляре
- Съемка производится горизонтально с допустимым отклонением (возможна ошибка до 10°)
- Высота съемки едина для всех фотографий (допускается отклонение до 3 см)

#### Требования к поверхности
- Ровная однородная горизонтальная поверхность
- Единая для всех фотографий

#### Требования к предметам
- Каждый предмет распологается на чистом листе бумаги
- Предмет или их набор находятся в центре листа и не выходят за его пределы
- Края данного листа видны на фотографии и не сливаются с поверхностью
- Предметы в наборе не перекрывают друг друга и не имеют общих границ
- Края предметов четко отличимы друг от друга, поверхности и листа бумаги (не менее 5 мм расстояния между предметами)
- Предметы не отбрасывают заметной тени
- Расположены в одной и тоже ориентации

#### Требование к многоугольнику
- Задается на отдельном листе бумаги черным маркером
- Данный лист фотографируется вместе с набором предметов для задачи
- Набор предметов и рассматриваемый для них многоугольник не пересекаются
- Толщина линии маркера до 5 мм
- Нарисован в центре листа бумаги

## Набор данных
[Датасет](https://drive.google.com/drive/folders/1S9s03F0Fk_Z-EFmSU9u3fNcNZpPfgtCL?usp=sharing)