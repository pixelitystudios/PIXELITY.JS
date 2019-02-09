import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

class FeatureExtraction{
	constructor(config){
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
    
	async predict(file){
        const tensor = await this.loadImage(file)
        const logits = this.mobilenet.infer(tensor, 'conv_preds')
        const prediction = await this.classifier.predictClass(logits);
        return {
            className: this.classes[prediction.classIndex],
            classIndex: prediction.classIndex
        }
	}

    async fetch(file){
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

    async processWeights(data, shape){
        if(data){
            let tensors = {}
            for(let d of data){
                let arr = []
                for(let tensor of Object.keys(d.tensor)){
                    arr.push(d.tensor[tensor])
                }
                tensors[d.label] = tf.tensor(arr, shape)
            }
            return tensors;
        }
    }

    async processManifest(manifestPath){
        const fetch = await this.fetch(manifestPath)
        return await this.processWeights(fetch.data, fetch.shape);
    }

	async load(manifestPath){
        this.classifier = knnClassifier.create()
        this.mobilenet = await mobilenetModule.load()
        this.manifest = await this.processManifest(manifestPath + '/manifest.json')
        this.classes = await this.fetch(manifestPath + '/classes.json')
        this.classifier.setClassifierDataset(this.manifest)
    }
}

export default FeatureExtraction;