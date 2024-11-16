// predict100.js

const CONFIG = {
    modelPath: './model_kerasnative_v4/model.json',
    defaultImagePath: './assets/samplepic.jpg',
    logoPath: './img/logotipo.png',
    inputShape: {
        height: 112,  // Ajustado según el modelo
        width: 112,   // Ajustado según el modelo
        channels: 3
    },
    maxRetries: 3,
    retryDelay: 2000
};

let model = null;

// Función para preprocesar la imagen según las especificaciones del modelo
function preprocessImage(img) {
    return tf.tidy(() => {
        // Convertir la imagen a tensor
        const tensor = tf.browser.fromPixels(img)
            // Redimensionar a 112x112 según el modelo
            .resizeNearestNeighbor([CONFIG.inputShape.height, CONFIG.inputShape.width])
            // Normalizar valores a [0,1]
            .toFloat()
            .div(255.0);
        
        // Expandir dimensiones para batch
        return tensor.expandDims(0);
    });
}

async function loadModelWithRetry(maxRetries = CONFIG.maxRetries) {
    let attempt = 0;
    
    while (attempt < maxRetries) {
        try {
            console.log(`Intentando cargar el modelo (intento ${attempt + 1}/${maxRetries})...`);
            $('.progress-bar').text(`Cargando modelo (intento ${attempt + 1}/${maxRetries})...`);
            
            // Definir la forma de entrada explícitamente
            const inputShape = [CONFIG.inputShape.height, CONFIG.inputShape.width, CONFIG.inputShape.channels];
            
            // Cargar el modelo con la configuración de entrada específica
            const loadedModel = await tf.loadLayersModel(CONFIG.modelPath, {
                onProgress: (fraction) => {
                    $('.progress-bar').text(`Cargando: ${(fraction * 100).toFixed(1)}%`);
                }
            });

            // Asegurarse de que el modelo tenga la forma de entrada correcta
            if (!loadedModel.inputs[0].shape.some(dim => dim === null)) {
                const newInputLayer = tf.layers.input({
                    shape: inputShape,
                    batchSize: null,
                    name: 'input_1'
                });
                
                // Reconstruir el modelo con la nueva capa de entrada
                const outputs = loadedModel.layers[1].apply(newInputLayer);
                const newModel = tf.model({ inputs: newInputLayer, outputs: outputs });
                
                return newModel;
            }
            
            console.log('Modelo cargado exitosamente');
            return loadedModel;
        } catch (error) {
            attempt++;
            console.error(`Intento ${attempt} fallido:`, error);
            
            if (attempt === maxRetries) {
                throw new Error(`Fallo al cargar el modelo después de ${maxRetries} intentos`);
            }
            await new Promise(resolve => setTimeout(resolve, CONFIG.retryDelay));
        }
    }
}

async function predict() {
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

        // Pre-procesar la imagen con las dimensiones correctas
        let tensor = preprocessImage(image);

        // Realizar predicción
        const predictions = await model.predict(tensor).data();
        tensor.dispose();

        // Mostrar resultados
        displayPredictions(predictions);

    } catch (error) {
        console.error('Error al realizar la predicción:', error);
        $("#prediction-list").html(`<div class="error">Error en la predicción: ${error.message}</div>`);
    }
}

function displayPredictions(predictions) {
    const top6 = Array.from(predictions)
        .map((probability, index) => ({
            probability: probability,
            className: TARGET_CLASSES[index] || `Clase ${index}`
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 6);

    $("#prediction-list").empty();
    top6.forEach(prediction => {
        $("#prediction-list").append(`
            <div class="prediction-item">
                ${prediction.className}: ${(prediction.probability * 100).toFixed(2)}%
            </div>
        `);
    });
}

async function initialize() {
    try {
        $('.progress-bar').show().text('Iniciando carga del modelo...');
        
        // Cargar el modelo
        model = await loadModelWithRetry();
        
        // Verificar que el modelo tenga la función predict
        if (typeof model.predict !== 'function') {
            throw new Error('La función predict no está definida en el modelo cargado');
        }
        
        // Configurar imagen por defecto
        const imageElement = $("#selected-image");
        imageElement.attr("src", CONFIG.defaultImagePath)
            .on('load', function() {
                // Realizar predicción inicial cuando la imagen esté cargada
                predict();
            })
            .on('error', function() {
                console.error('Error al cargar la imagen por defecto');
                $(this).attr('src', CONFIG.defaultImagePath);
            });
        
        $('.progress-bar').hide();
        
    } catch (error) {
        console.error('Error durante la inicialización:', error);
        $('.progress-bar').text(`Error: ${error.message}`).addClass('error');
    }
}

// Event listeners
$("#image-selector").change(function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            $("#selected-image").attr("src", e.target.result);
            $("#prediction-list").empty();
            // Esperar a que la imagen se cargue antes de predecir
            setTimeout(predict, 500);
        };
        reader.readAsDataURL(file);
    }
});

$("#predict-button").click(predict);

// Inicializar cuando el documento esté listo
$(document).ready(initialize);



