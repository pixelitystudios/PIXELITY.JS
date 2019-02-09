import * as tf from '@tensorflow/tfjs';
import imagenetClassesEs from './imagenet_classes_es.json'
import imagenetClassesEn from './imagenet_classes_en.json'
import distance from 'euclidean-distance'

class ImageClassification{
	constructor(config){
		this.config = {
			models: {
				mobilenet: 'http://localhost:8081/models/mobilenet/model.json',
				pixelity: ''
			},
			lang: 'es'
		}
		this.images = {
			xs: [],
			ys: []
		}
		this.xs;
		this.stopTraining = false;
		this.metadata = {
			xs: [],
			ys: [],
			images: {},
			history: []
		}
		if(config){
			for(let key of Object.keys(config)){
				if(this.config[key]){
					this.config[key] = config[key];
				}
			}
		}
	}

	loadImage(src) {
		return new Promise((resolve, reject) => {
			const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d')
			const img = new Image();
			img.src = src;
			img.onload = () => {
				ctx.drawImage(img, 0, 0)
				resolve(tf.fromPixels(canvas))
			};
		  	img.onerror = (err) => reject(err);
		});
	}

	async addExample(image, label){
		this.images.xs.push(image)
		this.images.ys.push(label)
		this.metadata.images = this.images;
	}

	async addImage(image, label){
		return new Promise(async (resolve, reject) => {
			this.images.xs.push(image)
			this.images.ys.push(label)
			this.metadata.images = this.images;
			this.extractFeatures(image).then((xs) => {
				this.metadata.xs.push(xs.dataSync())
				resolve()
			})
		})
	}

	downloadObjectAsJson(exportObj, exportName){
		var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 4));
		var downloadAnchorNode = document.createElement('a');
		downloadAnchorNode.setAttribute("href",     dataStr);
		downloadAnchorNode.setAttribute("download", exportName + ".json");
		document.body.appendChild(downloadAnchorNode); // required for firefox
		downloadAnchorNode.click();
		downloadAnchorNode.remove();
	}

	save(){
		return new Promise(async (resolve, reject) => {
			this.downloadObjectAsJson(this.metadata, 'metadata')
			this.model.save('downloads://model')
			resolve()
		})
	}
	

	cropImage(img) {
		const size = Math.min(img.shape[0], img.shape[1]);
		const centerHeight = img.shape[0] / 2;
		const beginHeight = centerHeight - (size / 2);
		const centerWidth = img.shape[1] / 2;
		const beginWidth = centerWidth - (size / 2);
		return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
	}

	resizeImage(image) {
		return tf.image.resizeBilinear(image, [224, 224]);
	}

	batchImage(image) {
		// Expand our tensor to have an additional dimension, whose size is 1
		const batchedImage = image.expandDims(0);
	  
		// Turn pixel data into a float between -1 and 1.
		return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
	}

	useCustomModel(){
		this.customModel = true;
	}

	loadAndProcessImage(image) {
		const croppedImage = this.cropImage(image);
		const resizedImage = this.resizeImage(croppedImage);
		const batchedImage = this.batchImage(resizedImage);
		return batchedImage;
	}

	extractFeatures(image){
		return new Promise((resolve, reject) => {
			this.loadImage(image).then(img => {
				const processedImage = this.loadAndProcessImage(img);
				if(this.customModel){
					const activatedImage = this.pretrainedModel.predict(processedImage);
					resolve(activatedImage)
					processedImage.dispose();
				}else{
					resolve(processedImage)
				}
			})
		})
	}

	getModel(numberOfClasses) {
		const model = tf.sequential({
		  	layers: [
				tf.layers.flatten({inputShape: [7, 7, 256]}),
				tf.layers.dense({
					units: 100,
					activation: 'relu',
					kernelInitializer: 'varianceScaling',
					useBias: true
				}),
				tf.layers.dense({
					units: numberOfClasses,
					kernelInitializer: 'varianceScaling',
					useBias: false,
					activation: 'softmax'
				})
		  	],
		});
	  
		model.compile({
			optimizer: tf.train.adam(0.0001),
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy'],
		});
	  
		return model;
	}

	async train(conf){
		return new Promise(async (resolve, reject) => {
			let config = {
				epochs: 20,
				shuffle: true,
				learningRate: 0.001,
				onEpochEnd: (epoch, log) => {
					console.log(`Epoch: ${epoch}, loss: ${log.loss}, accuracy: ${log.acc}`)
				}
			}
			if(conf){
				for(let key of Object.keys(conf)){
					if(config[key]){
						config[key] = config[key];
					}
				}
			}
			const layer = this.mobilenet.getLayer('conv_pw_13_relu');
			this.pretrainedModel = tf.model({inputs: this.mobilenet.inputs, outputs: layer.output});
			const xs = await this.loadImages(this.images.xs, this.pretrainedModel);
			this.metadata.xs = xs.dataSync();
			const ys = this.addLabels(this.images.ys)
			this.metadata.ys = ys.dataSync()
			this.model = this.getModel(this.images.ys.length)
			this.model.fit(xs, ys, {
				epochs: config.epochs,
				shuffle: config.shuffle,
				callbacks: {
					onEpochEnd: async (epoch, log) => {
						this.metadata.history.push({
							epoch: epoch,
							log: log
						})
						config.onEpochEnd(epoch, log)
						await tf.nextFrame()
						if(this.stopTraining){
							this.model.stopTraining = true;
							resolve()
						}
						if(epoch == config.epochs - 1){
							resolve()
						}
						if(log.loss < config.learningRate){
							this.stopTraining = true;
							resolve()
						}
					}
				}
			})
		})
	}

	getLabelsAsObject(labels) {
		let labelObject = {};
		for (let i = 0; i < labels.length; i++) {
		  	const label = labels[i];
		  	if (labelObject[label] === undefined) {
				// only assign it if we haven't seen it before
				labelObject[label] = Object.keys(labelObject).length;
		  	}
		}
		return labelObject;
	}

	addLabels(labels) {
		return tf.tidy(() => {
		  	const classes = this.getLabelsAsObject(labels);
		  	const classLength = Object.keys(classes).length;
	  
		  	let ys;
		  	for (let i = 0; i < labels.length; i++) {
				const label = labels[i];
				const labelIndex = classes[label];
				const y = this.oneHot(labelIndex, classLength);
				if (i === 0) {
			  		ys = y;
				} else {
			  		ys = ys.concat(y, 0);
				}
		  	}
		  	return ys;
		});
	}

	oneHot(labelIndex, classLength) {
		return tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength));
	}

	loadDataset(dataset){

	}

	loadImages(images, model) {
		let promise = Promise.resolve();
		for (let i = 0; i < images.length; i++) {
		  	const image = images[i];
		  	promise = promise.then(data => {
			return this.loadImage(image).then(loadedImage => {
				// Note the use of `tf.tidy` and `.dispose()`. These are two memory management
				// functions that Tensorflow.js exposes.
				// https://js.tensorflow.org/tutorials/core-concepts.html
				//
				// Handling memory management is crucial for building a performant machine learning
				// model in a browser.
				return tf.tidy(() => {
					const processedImage = this.loadAndProcessImage(loadedImage);
					const prediction = model.predict(processedImage);
		
					if (data) {
						const newData = data.concat(prediction);
						data.dispose();
						return newData;
					}
		
						return tf.keep(prediction);
					});
				});
		  	});
		}
		return promise;
	}

	async predict(features){
		if(this.customModel){
			const prediction = await this.model.predict(features);
			const labelPrediction = prediction.as1D().argMax().dataSync()[0]
			const result = {
				index: labelPrediction,
				label: this.metadata.ys[labelPrediction]
			}		
			return result;
		}else{
			const prediction = await this.mobilenet.predict(features);
			const labelPrediction = prediction.as1D().argMax().dataSync()[0]
			const result = {
				index: labelPrediction,
				label: (this.config.lang == 'es') ? imagenetClassesEs[labelPrediction] : imagenetClassesEn[labelPrediction]
			}		
			return result;
		}
	}


	async fetch(file, cache = false){
        return new Promise((resolve, reject) => {
			var xmlHttp = new XMLHttpRequest();
			xmlHttp.onreadystatechange = function() { 
				if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
					resolve(JSON.parse(xmlHttp.responseText));
				}
			}
			xmlHttp.open("GET", file, true); // true for asynchronous 
			xmlHttp.send(null);
        })
    }

	async loadModel(model){
		this.mobilenet = await tf.loadModel(this.config.models.mobilenet)
		const layer = this.mobilenet.getLayer('conv_pw_13_relu');
		this.pretrainedModel = tf.model({inputs: this.mobilenet.inputs, outputs: layer.output});
		if(model){
			this.model = await tf.loadModel(tf.io.browserHTTPRequest(model + 'model.json', {method: 'GET'}));
        	this.metadata = await this.fetch(model + 'metadata.json', true);
		}
	}
}

export default ImageClassification;