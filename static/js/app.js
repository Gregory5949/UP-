function getData(data) {
    const obj = {
        age_group: {
            type: 'doughnut',
            label: 'Возраст',
            labels: ['<26', '26-55', '55+'],
            backgroundColor: ['#FF4D4D', '#ffc000', '#66B3FF'],
            title: 'Возрастная группа',
        },
        gender: {
            type: 'doughnut',
            label: 'Пол',
            labels: ['Мужской', 'Женский'],
            backgroundColor: ['#FF4D4D', '#66B3FF'],
            title: 'Пол',
        },
        day_of_week: {
            type: 'bar',
            label: 'Дни недели',
            labels: ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
            backgroundColor: '#ffc000',
            title: 'Активность по дням недели'
        },
        time_of_day: {
            type: 'bar',
            label: 'Время суток',
            labels: ['Ночь', 'Утро', 'День', 'Вечер'],
            backgroundColor: '#ffc000',
            title: 'Активность по времени суток'
        }
    };
    for (const key in data)
        obj[key].data = data[key].data;
    return obj;
}

function createCharts(data) {
    let content = document.getElementById('content__inner');

    for (const key in data) {
        let canvas = document.createElement('canvas');
        canvas.className = 'card';

        new Chart(canvas, {
            type: data[key].type,
            data: {
                labels: data[key].labels,
                datasets: [{
                    label: data[key].label,
                    data: data[key].data,
                    backgroundColor: data[key].backgroundColor,
                    borderColor: '#000',
                    borderWidth: 1.5
                }]
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: data[key].title,
                        padding: {
                            top: 10,
                            bottom: 30
                        },
                        font: {
                            size: 18
                        }
                    }
                },
            }
        });
        content.appendChild(canvas);
    }
}

function clearModal() {
    let drop = document.getElementById('myDropzone');
    let classList = Array.from(drop.classList).slice(0);

    for (const className of classList)
        if(className != 'dropzone' && className != 'dz-clickable')
            drop.classList.remove(className)
    if(drop.childNodes.length > 1)
        while (drop.childNodes.length > 1)
            drop.removeChild(drop.childNodes[1]);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Модальное окно
document.getElementById('upload__btn').addEventListener('click',function (e) {
    document.getElementById('modal').classList.add('active');
});

// Обработка drop
document.getElementById('myDropzone').addEventListener('drop', async function () {
    let response = await fetch('/upload', {
        "method": "POST",
        "headers": {"Content-Type": "application/json"}
    });

    if (response.ok) {
        let json = await response.json();
        let data = getData(json);
        document.getElementById('content__inner').replaceChildren();
        createCharts(data);
        document.getElementById('modal').classList.remove('active');
        clearModal();
    }
});


// Сохранить в pdf
document.getElementById('download__btn').addEventListener('click', function (e) {
    let content = document.getElementById('content__inner');
    let opt = {
        margin: 0,
        filename: 'myfile.pdf',
        image: {type: 'jpeg', quality: 0.98},
        html2canvas: {scale: 5},
        jsPDF: {unit: 'in', format: 'a2', orientation: 'portrait'}
    };

    html2pdf().set(opt).from(content).save();
});