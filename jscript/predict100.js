// Configuración de rutas ajustada a tu estructura
const CONFIG = {
    modelPath: './model_kerasnative_v4/model.json',
    defaultImagePath: './assets/samplepic.jpg',
    logoPath: './img/logotipo.png'
};

// Variable global para el modelo
let model = null;

// Función para verificar si un archivo existe
async function checkFileExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        return response.ok;
    } catch (error) {
        console.error(`Error checking file ${url}:`, error);
        return false;
    }
}

// Función para cargar el modelo con reintentos
async function loadModelWithRetry(maxRetries = 3) {
    let attempt = 0;
    
    while (attempt < maxRetries) {
        try {
            // Verificar si el archivo model.json existe
            const modelExists = await checkFileExists(CONFIG.modelPath);
            if (!modelExists) {
                throw new Error(`Archivo del modelo no encontrado en: ${CONFIG.modelPath}`);
            }

            // Mostrar progreso de carga
            $('.progress-bar').text(`Cargando modelo (intento ${attempt + 1}/${maxRetries})...`);
            
            // Cargar el modelo
            const loadedModel = await tf.loadLayersModel(CONFIG.modelPath, {
                onProgress: (fraction) => {
                    $('.progress-bar').text(`Cargando modelo: ${(fraction * 100).toFixed(1)}%`);
                }
            });
            
            // Calentar el modelo
            const dummyInput = tf.zeros([1, 224, 224, 3]);
            await loadedModel.predict(dummyInput).data();
            dummyInput.dispose();
            
            return loadedModel;
        } catch (error) {
            attempt++;
            console.error(`Intento ${attempt} fallido:`, error);
            
            if (attempt === maxRetries) {
                throw new Error(`No se pudo cargar el modelo después de ${maxRetries} intentos`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }
}

// Función de inicialización mejorada
async function initialize() {
    try {
        $('.progress-bar').show().text('Iniciando carga del modelo...');
        
        // Cargar el modelo con reintentos
        model = await loadModelWithRetry();
        
        // Configurar la imagen por defecto
        const imageElement = $("#selected-image");
        imageElement.attr("src", CONFIG.defaultImagePath)
            .on('error', function() {
                console.error('Error al cargar la imagen por defecto');
                $(this).attr('src', './assets/samplepic.jpg'); // Imagen de respaldo
            });
        
        // Configurar el logo
        $("#logo").attr("src", CONFIG.logoPath)
            .on('error', function() {
                console.warn('Error al cargar el logo');
                $(this).hide(); // Ocultar si no se puede cargar
            });
        
        $('.progress-bar').hide();
        console.log('Inicialización completada con éxito');
        
        // Realizar predicción inicial
        predictOnLoad();
        
    } catch (error) {
        console.error('Error durante la inicialización:', error);
        $('.progress-bar').text(`Error: ${error.message}`).addClass('error');
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

// Manejador del botón de predicción
$("#predict-button").click(async function() {
    if (!model) {
        console.error('El modelo no está cargado');
        $("#prediction-list").html('<div class="error">El modelo no está cargado correctamente</div>');
        return;
    }

    try {
        const image = $('#selected-image').get(0);
        
        if (!image || !image.complete) {
            throw new Error('La imagen no está completamente cargada');
        }

        // Pre-procesar la imagen
        let tensor = tf.tidy(() => {
            image.width = 600;
            image.height = 450;

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
        $("#prediction-list").html(`<div class="error">Error al realizar la predicción: ${error.message}</div>`);
    }
});

// Inicializar la aplicación
initialize();



