//@ts-check
import * as tf from '@tensorflow/tfjs';
import * as jpeg from 'jpeg-js'

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadDecapitatedMobilenet() {
    const mobilenet = await tf.loadModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json"
    );

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer("conv_pw_13_relu");
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

class ImageClassifier {
    constructor() {
        this.currentModelPath = null;
        this.decapitatedMobilenet = null;
        this.model = null;
        this.labels = null;
    }

    async init() {
        this.decapitatedMobilenet = await loadDecapitatedMobilenet();
    }

    async loadImage(url){
        return new Promise((resolve, reject) => {
            var canvas = document.createElement('canvas');
            var context = canvas.getContext('2d');
            var img = new Image();
            img.onload = function(){
                context.drawImage(this, this.width, this.height);
                resolve(context.getImageData(0, 0, this.width, this.height))
            }
            img.crossOrigin = 'Anonymous';
            img.src = url;
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
    
	loadAndProcessImage(image) {
		const croppedImage = this.cropImage(image);
		const resizedImage = this.resizeImage(croppedImage);
		const batchedImage = this.batchImage(resizedImage);
		return batchedImage;
	}

	extractFeaturesData(image){
		return new Promise((resolve, reject) => {
			this.loadImageData(image).then(img => {
				const processedImage = this.loadAndProcessImage(img);
                resolve(processedImage)
			})
		})
	}

    loadImageData(src) {
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

    imageByteArray(image, numChannels) {
        const pixels = image.data;
        const numPixels = image.width * image.height;
        const values = new Int32Array(numPixels * numChannels);

        for (let i = 0; i < numPixels; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                values[i * numChannels + channel] = pixels[i * 4 + channel];
            }
        }

        return values;
    }

    imageToTensor(image, numChannels) {
        const values = this.imageByteArray(image, numChannels);
        const outShape = [1, image.height, image.width, numChannels];
        return tf
            .tensor4d(values, outShape, "int32")
            .toFloat()
            .resizeBilinear([224, 224])
            .div(tf.scalar(127))
            .sub(tf.scalar(1));
    }

    extractFeatures(image){
		return new Promise((resolve, reject) => {
            const tensor = this.imageToTensor(image, 3)
            resolve(tensor)
		})
	}

    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    buildRetrainingModel(denseUnits, numClasses, learningRate) {
        this.model = tf.sequential({
            layers: [
                // Flattens the input to a vector so we can use it in a dense layer. While
                // technically a layer, this only performs a reshape (and has no training
                // parameters).
                tf.layers.flatten({
                    inputShape: this.decapitatedMobilenet.outputs[0].shape.slice(
                        1
                    )
                }),
                // Layer 1.
                tf.layers.dense({
                    units: denseUnits,
                    activation: "relu",
                    kernelInitializer: "varianceScaling",
                    useBias: true
                }),
                // Layer 2. The number of units of the last layer should correspond
                // to the number of classes we want to predict.
                tf.layers.dense({
                    units: numClasses,
                    kernelInitializer: "varianceScaling",
                    useBias: false,
                    activation: "softmax"
                })
            ]
        });

        // Creates the optimizers which drives training of the model.
        const optimizer = tf.train.adam(learningRate);
        // We use categoricalCrossentropy which is the loss function we use for
        // categorical classification which measures the error between our predicted
        // probability distribution over classes (probability that an input is of each
        // class), versus the label (100% probability in the true class)>
        this.model.compile({
            optimizer: optimizer,
            loss: "categoricalCrossentropy"
        });
    }

    currentModelPath() {
        return this.currentModelPath;
    }

    predict(x) {
        // Assume we are getting the embeddings from the decapitatedMobilenet
        let embeddings = x;
        // If the second dimension is 224, treat it as though it's an image tensor
        if (x.shape[1] === 224) {
            embeddings = this.decapitatedMobilenet.predict(x);
        }

        let { values, indices } = this.model.predict(embeddings).topk();
        return {
            label: this.labels.Labels[indices.dataSync()[0]],
            confidence: values.dataSync()[0]
        };
    }

    async loadModel(dirPath) {
        this.model = await tf.loadModel(dirPath + "/model.json");
        this.labels = await this.fetch(dirPath + "/labels.json")

        this.currentModelPath = dirPath;
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
}

export default ImageClassifier;
