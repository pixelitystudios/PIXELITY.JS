import * as faceapi from './face-api'
import * as tf from '@tensorflow/tfjs'
import Emitter from '../Utils/event-emitter'

class FaceID{
    constructor(path, models){
        this.MODEL_URL = path
        this.faces = {}
        this.faceMatcher = null;
        this.model = faceapi
        this.debug = false;
        this.error = false;
        this.errorMessage = {error: false, message: ''}
        this.StopTraining = false;
        this.labeledDescriptors = []
        this.observable = new Emitter()
        this.mtcnnParams = {
            minFaceSize: 200
        }
        this.loadModels(models); 
    }

    on(event, listener){
        this.observable.on(event, listener);
    }

    async loadModels(models){
        await faceapi.loadSsdMobilenetv1Model(this.MODEL_URL)
        await faceapi.loadMtcnnModel(this.MODEL_URL)
        await faceapi.loadFaceLandmarkModel(this.MODEL_URL)
        await faceapi.loadFaceRecognitionModel(this.MODEL_URL)
        this.observable.emit('model:load');
    }

    async log(message){
        if(this.debug){
            console.log(message)
        }
    }

    async fromVideo(input){
        return new Promise(async (resolve, reject) => {
            const options = new faceapi.MtcnnOptions(this.mtcnnParams)
            const fullFaceDescriptions = await faceapi.detectAllFaces(input, options).withFaceLandmarks().withFaceDescriptors()
            
            resolve(fullFaceDescriptions)
        })
    }

    setDebug(debug){
        this.debug = debug;
    }

    async processWeights(data){
        if(data){
            let w = [];
            for(let face of data){
                let ww = []
                for(let key of Object.keys(face)){
                    ww.push(face[key])
                }
                w.push(new Float32Array(ww))
            }
            return w
        }
    }

    async addFace(image, label){
        if(!this.faces[label]){
            this.faces[label] = {
                descriptors: [],
                images: [],
                labeled: null
            }
        }
        this.faces[label].images.push(image)
    }

    async getManifest(label){
        return new Promise((resolve, reject) => {
            if(this.faces[label]){
                resolve(this.faces[label].descriptors)
            }
        })
    }

    async nextFace(index){
        let faces = Object.keys(this.faces)
        let face = faces[index];
        if(face){
            let images = this.faces[face].images;
            let label = face;
            let descriptors = await this.ProcessImages(images)
            let interval = setInterval(() => {
                if(this.error){
                    clearInterval(interval)
                    return false;
                }
                if(descriptors.length == images.length){
                    this.faces[face].descriptors = descriptors;
                    this.labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label,descriptors))
                    this.nextFace(index + 1)
                    clearInterval(interval)
                }
                
            }, 100)
        }else{
            this.StopTraining = true;
        }
    }

    async train(){
        this.StopTraining = false;
        return new Promise(async (resolve, reject) => {
            this.nextFace(0);
            let interval = setInterval(async () => {
                if(this.error){
                    clearInterval(interval)
                    resolve(this.errorMessage)
                    return false;
                }
                if(this.StopTraining){
                    this.faceMatcher = await new faceapi.FaceMatcher(this.labeledDescriptors)
                    resolve({error: false})
                    clearInterval(interval)
                }
            }, 1000)
        })
    }

    async ProcessImages(images){
        let descriptors = []
        images.map(async im => {
            const referenceImage = await faceapi.fetchImage(im)
            const fullFaceDescription = await faceapi.detectSingleFace(referenceImage).withFaceLandmarks().withFaceDescriptor()
    
            if(fullFaceDescription) {
                descriptors.push(fullFaceDescription.descriptor)
            }else{
                this.error = true;
                this.errorMessage = {
                    error: true,
                    message: 'Cant create face model from ' + im + ' source'  
                }
            }
        })
        return descriptors;
    }

    async recognition(data){
        return new Promise(async (resolve, reject) => {
            const labeledDescriptors = []
            let FaceMatcher;
            for(let face of data){


                labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(
                    face.label,
                    await this.processWeights(face.descriptors)
                ))

            }
            if(!labeledDescriptors){
                return "No se encontraron caras en este modelo"
            }
            // create FaceMatcher with automatically assigned labels
            // from the detection results for the reference image
            FaceMatcher = new faceapi.FaceMatcher(labeledDescriptors)

            resolve(FaceMatcher);
        })
    }

    async createModel(canvas){
        return await faceapi.detectSingleFace(canvas).withFaceLandmarks().withFaceDescriptor()
    }
}

export default FaceID;