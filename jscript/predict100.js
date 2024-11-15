function simulateClick(tabID) {
    document.getElementById(tabID).click();
}

function predictOnLoad() {
    // Simular un clic en el botón predecir
    setTimeout(simulateClick.bind(null,'predict-button'), 500);
};

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
    
    // Simular un clic en el botón predecir.
    // Esto introduce un retardo de 0,5 segundos antes del clic.
    // Sin este retardo, el modelo se carga pero no puede predecir automáticamente.
    setTimeout(simulateClick.bind(null,'predict-button'), 500);
});

let model;
(async function () {
    // Cargar el modelo
    model = await tf.loadLayersModel('model_kerasnative_v4/model.json');
    $("#selected-image").attr("src", "assets/samplepic.jpg");
    
    // Ocultar el spinner de carga del modelo
    $('.progress-bar').hide();
    
    // Simular un clic en el botón predecir
    predictOnLoad();
})();

// Cree el método asincrónico #predict-button donde se realiza el pre-procesamiento de las 
// imágenes a tamaño 244x244 para pasarlo como tensor al modelo generado, retornando 
// la probabilidad ordenada por las 6 clases (TARGET_CLASSES[i])
$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);

    // Pre-procesar la imagen
    image.width = 600;  // Ajuste del tamaño de la imagen si es necesario
    image.height = 450;

    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])  // Redimensionar la imagen a 224x224
        .toFloat();  // Convertir a tipo flotante

    // Normalización: Convertir los valores de 0-255 a un rango [-1, 1] o [0, 1]
    const normalizedTensor = tensor.div(tf.scalar(127.5)).sub(tf.scalar(1));  // Si el modelo fue entrenado con esta normalización

    // Pasa el tensor al modelo y llama a predecir sobre él.
    // Predecir devuelve un tensor.
    // data() carga los valores del tensor de salida y devuelve
    // un array tipado cuando el cálculo se ha completado.

    let predictions = await model.predict(normalizedTensor.expandDims(0)).data();
    
    // Ordenar las predicciones y tomar las 6 más probables
    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: TARGET_CLASSES[i]  // estamos seleccionando el valor del objeto
            };
        })
        .sort(function (a, b) {
            return b.probability - a.probability;
        })
        .slice(0, 6);  // Ajuste aquí el número de predicciones de salida.

    // Vaciar la lista de predicciones
    $("#prediction-list").empty();

    // Mostrar las predicciones en la interfaz
    top5.forEach(function (p) {
        $("#prediction-list").append(<li>${p.className}: ${p.probability.toFixed(6)}</li>);
    });
});





