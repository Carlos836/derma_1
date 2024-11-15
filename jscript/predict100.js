// Definir las clases objetivo (ajusta según tus clases reales)
const TARGET_CLASSES = {
    0: "Clase 1",
    1: "Clase 2",
    2: "Clase 3",
    3: "Clase 4",
    4: "Clase 5",
    5: "Clase 6"
};

// Variable global para el modelo
let model = null;

// Función para simular click
function simulateClick(tabID) {
    const element = document.getElementById(tabID);
    if (element) {
        element.click();
    }
}

// Función para predecir al cargar
function predictOnLoad() {
    if (model) {
        setTimeout(() => simulateClick('predict-button'), 500);
    }
}

// Manejador de cambio de imagen
$("#image-selector").change(function() {
    const reader = new FileReader();
    reader.onload = function() {
        const dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    };

    const file = $("#image-selector").prop('files')[0];
    if (file) {
        reader.readAsDataURL(file);
        setTimeout(() => simulateClick('predict-button'), 500);
    }
});

// Función de inicialización
async function initialize() {
    try {
        // Especificar la forma de entrada del modelo
        const modelConfig = {
            inputs: [{
                name: 'input_1',
                shape: [null, 224, 224, 3],
                dtype: 'float32'
            }]
        };

        // Cargar el modelo con la configuración
        model = await tf.loadLayersModel('model_kerasnative_v4/model.json', modelConfig);
        
        // Configurar la imagen por defecto
        $("#selected-image").attr("src", "assets/samplepic.jpg");
        
        // Ocultar la barra de progreso
        $('.progress-bar').hide();
        
        // Realizar predicción inicial
        predictOnLoad();
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        $('.progress-bar').text('Error al cargar el modelo: ' + error.message);
    }
}

// Manejador del botón de predicción
$("#predict-button").click(async function() {
    if (!model) {
        console.error('El modelo no está cargado');
        return;
    }

    try {
        const image = $('#selected-image').get(0);
        
        // Verificar que la imagen está cargada
        if (!image || !image.complete) {
            console.error('La imagen no está completamente cargada');
            return;
        }

        // Pre-procesar la imagen
        let tensor = tf.tidy(() => {
            // Establecer dimensiones de visualización
            image.width = 600;
            image.height = 450;

            // Crear y pre-procesar el tensor
            return tf.browser.fromPixels(image)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });

        // Realizar predicción
        const predictions = await model.predict(tensor).data();
        tensor.dispose();

        // Procesar resultados
        const top6 = Array.from(predictions)
            .map((probability, index) => ({
                probability: probability,
                className: TARGET_CLASSES[index]
            }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 6);

        // Mostrar predicciones
        $("#prediction-list").empty();
        top6.forEach(prediction => {
            $("#prediction-list").append(`
                <div class="prediction-item">
                    ${prediction.className}: ${prediction.probability.toFixed(6)}
                </div>
            `);
        });

    } catch (error) {
        console.error('Error al realizar la predicción:', error);
        $("#prediction-list").append(`<div>Error al realizar la predicción: ${error.message}</div>`);
    }
});

// Inicializar la aplicación
initialize();




