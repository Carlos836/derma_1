// Después de cargar el modelo queremos hacer una predicción sobre la imagen por defecto.
// Así, el usuario verá las predicciones cuando se cargue la página por primera vez.

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
	
	model = await tf.loadModel('model_kerasnative_v4/model.json');
	$("#selected-image").attr("src", "assets/samplepic.jpg")
	
	
	
	// Ocultar el spinner de carga del modelo
	$('.progress-bar').hide();
	
	// Simular un clic en el botón predecir
	predictOnLoad();
	
	
})();


// Cree el método asincronico #predict-button donde se realice el pre-procesameinto de las 
// imagenes a tamaño 244X244 para pasarlo como tensor al modelo generado retornando 
// la probabilidad ordenada por las 6 clases (TARGET_CLASSES[i])


$("#predict-button").click(async function () {
	
	
	
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	image.width = 600;
  	image.height = 450;

	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
	.toFloat();
	
	
	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();

	// Pasa el tensor al modelo y llama a predecir sobre él.
	// Predecir devuelve un tensor.
	// data() carga los valores del tensor de salida y devuelve
	// por medio de un array tipado cuando el cálculo se ha completado.

	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // este es el Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // estamos seleccionando el valor del objeto
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 6); // ajuste aquí el número de predicciones de salida.
	
	
$("#prediction-list").empty();
top5.forEach(function (p) {

	$("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);

	
	});
	
	
});









