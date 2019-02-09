import * as tf from '@tensorflow/tfjs'
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import nlp from 'lorca-nlp'
import shuffle from 'shuffle-array'
import moment from 'moment';

Array.prototype.extend = function (other_array) {
    /* You should include a test to check whether other_array really is an array */
    other_array.forEach(function(v) {this.push(v)}, this);
}
/**
 *
 *
 * @class TextClassifier
 */
class TextClassifier{
    /**
     *Creates an instance of TextClassifier.
     * @param {*} config
     * @memberof TextClassifier
     */
    constructor(config){
        this.metadata = {}
        this.metadata = {}
        this.words = []
        this.intents = []
        this.classes = []
        this.classifierClasses = []
        this.documents = []
        this.training = []
        this.output = []
        this.context = null
        this.train_x = []
        this.stopTraining = false;
        this.xs = []
        this.ys = []
        this.train_y = []
        this.model;
        this.inputModel;
        this.config = {
            batchSize: 8,
            epochs: 100,
            dropout: 0.25,
            type: 'rnn',
            learningRate: 0.0001,
            adamBeta1: 0.025,
            adamBeta2: 0.1,
            confidence: 0.51,
            onEpochEnd: (epoch, log) => {
                console.log({
                    epoch: epoch,
                    log: log
                })
            }
        }
        if(config){
            for(let c of Object.keys(config)){
                if(this.config[c]){
                    this.config[c] = config[c];
                }
            }
        }
        this.classifier = knnClassifier.create()
    }

    setContext(context){
        this.context = context;
    }

    getContext(){
        return this.context;
    }
    /**
     *
     *
     * @param {*} intents
     * @returns
     * @memberof TextClassifier
     */
    async processIntent(intents){
        let dataset = {
            intents: []
        }
        
        for(let n of intents){
            dataset.intents.push({
                tag: '_' + Math.random().toString(36).substr(2, 9),
                patterns: n.patterns,
                responses: n.responses,
                extras: n
            })
        }
        return dataset
    }

    async fit(){
        let prom = await Promise.all(this.train_x.map(async (x, i) => {
            await this.classifier.addExample(tf.tensor(x), i)
            await this.classifier.addExample(tf.tensor(x), i)
            await this.classifier.addExample(tf.tensor(x), i)
        }))
        console.log(this.metadata)
    }

    /**
     *
     *
     * @param {*} intents
     * @memberof TextClassifier
     */
    async compile(intents){
        if(typeof intents === 'string'){
            intents = await this.fetch(intents, false);
        }
        intents = await this.processIntent(intents);
        this.intents = intents;
        // loop through each sentense in our intents patterns
        intents.intents.map(async (intent) => await this.stemPattern(intent))
        // stem and lower each word
        let words = []
        this.words.map((w, index) => (w == '') ? null : words.push( nlp(w).stem(w.toLowerCase()).replace(/[?]/g, '')))
        // remove duplicates
        words = words.filter((item, pos) =>  words.indexOf(item) == pos)
        this.words = words;
        // remove duplicates
        this.classes = this.classes.filter((item, pos) => this.classes.indexOf(item) == pos);
        // training set, bag of words for each sentence
        this.documents.map((doc, index) => {
            // initialize our bag of words
            let bag = []
            // list of tokenized words for the pattern
            let pattern_words = doc[0]
            // stem each word
            pattern_words = pattern_words.map((word) => nlp(word).stem(word.toLowerCase()))
            // create our bag of words array
            words.map((w, i) => (pattern_words.includes(w)) ? bag.push(1) : bag.push(0))
            // create an empty array for our output
            let output_row = Array(this.classes.length).fill(0);
            // output is a '0' for each tag and '1' for current tag
            output_row[this.classes.indexOf(doc[1])] = 1
            // add our bag of words and output_row to our training list
            this.training.push([bag, output_row])
        })
        // shuffle our features and turn into np.array
        shuffle(this.training);
        this.training.map((item) => {
            this.train_x.push(item[0])
            this.train_y.push(item[1])
        })
        this.metadata.words = this.words
        this.metadata.documents = this.documents
        this.metadata.classes = this.classes
        this.metadata.shape = [this.train_x[0].length,this.train_y[0].length]
        this.metadata.intents = this.intents
        this.metadata.history = []
    }
    /**
     *
     *
     * @memberof TextClassifier
     */
    async train(){
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [this.train_x[0].length] }));
        model.add(tf.layers.dropout({rate: this.config.dropout}));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({rate: this.config.dropout}));
        model.add(tf.layers.dense({ units: this.train_y[0].length, activation: 'softmax' }));
        model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

        const xs = tf.tensor(this.train_x);
        const ys = tf.tensor(this.train_y);
        await model.fit(xs, ys, {
            epochs: this.config.epochs,
            batchSize: this.config.batchSize,
            shuffle: true,
            // verbose: 1,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    this.metadata.history.push({
                        epoch: epoch,
                        log: log
                    })
                    this.config.onEpochEnd(epoch, log)
                    await tf.nextFrame()
                    if(this.stopTraining){
                        model.stopTraining = true;
                    }
                    if(log.loss < this.config.learningRate){
                        this.stopTraining = true;
                    }
                }
            }
        })
        this.model = model;
        xs.dispose()
        ys.dispose()
    }
    /**
     *
     *
     * @param {*} input
     * @returns
     * @memberof TextClassifier
     */
    async prepareInput(input){
        let bag = Array(this.metadata.shape[0]).fill(0);
        let sentence = nlp(input).words().get();
        let words = []
        sentence.map((w, index) => words.push( nlp(w).stem(w.toLowerCase()).replace(/[?]/g, '')))
        this.metadata.words.map((word, index) => {
            words.map((w,i) => {
                if(w == word){
                    bag[index] = 1
                }
            })
        })
        return bag;     
    }
    /**
     *
     *
     * @param {*} sentence
     * @returns
     * @memberof TextClassifier
     */
    async predict(sentence){
        const bag = await this.prepareInput(sentence);
        let _this = this;
        return new Promise(async (resolve, reject) => {
            return await tf.tidy(()=>{
                //converter to tensor array
                let data = tf.tensor2d(bag, [1, bag.length]);

                if(this.config.type == 'rnn'){
                    //generate probabilities from the model
                    let predictions = this.model.predict(data).dataSync();
                    //filter out predictions below a threshold
                    let prediction = {
                        accuracy: 0,
                        intent: null
                    };
                    predictions.map((p, i) => {
                        if(p > _this.config.confidence){
                            let intent = _this.metadata.intents.intents.filter((c) => (_this.metadata.classes[i] == c.tag) ? c : null)[0]
                            if(_this.context){
                                if(intent.extras.context == _this.context){
                                    prediction.accuracy = p;
                                    prediction.intent = intent
                                }
                            }else{
                                prediction.accuracy = p;
                                prediction.intent = intent
                            }
                        }
                    })
                    resolve(prediction)
                }else{
                    _this.classifier.predictClass(data).then(predictions => {
                        let prediction = {
                            accuracy: 0,
                            intent: null
                        };
                        let className = _this.documents[predictions.classIndex][1]
                        let accuracy = predictions.confidences[predictions.classIndex]
                        console.log(className)
                        console.log(_this.metadata.intents.intents)
                        console.log(accuracy, _this.config.confidence)
                        if( accuracy > _this.config.confidence){
                            let intent = _this.metadata.intents.intents.filter((c) => (className == c.tag) ? c : null)[0]
                            if(_this.context){
                                if(intent.extras.context == _this.context){
                                    prediction.accuracy = accuracy;
                                    prediction.intent = intent
                                }
                            }else{
                                prediction.accuracy = accuracy;
                                prediction.intent = intent
                            }
                        }
                        resolve(prediction)
                    })
                }
            }) 
        })
    }
    /**
     *
     *
     * @param {*} text
     * @param {*} name
     * @memberof TextClassifier
     */
    downloadTextFile(text, name) {
        const a = document.createElement('a');
        const type = name.split(".").pop();
        a.href = URL.createObjectURL( new Blob([text], { type:`text/${type === "txt" ? "plain" : type}` }) );
        a.download = name;
        a.click();
    }
    /**
     *
     *
     * @param {*} config
     * @returns
     * @memberof TextClassifier
     */
    async save(config){
        if(config.type == 'download'){
            await this.model.save('downloads://model')
            this.downloadTextFile(JSON.stringify(this.metadata, null, 4), 'metadata.json')
        }else if(config.type == 'http'){
            await this.model.save(tf.io.browserHTTPRequest(config.savePathModel, {method: 'POST'}))
    
            var xhr = new XMLHttpRequest();
            var url = config.savePathMetadata;
            xhr.open("POST", url, true);
            xhr.setRequestHeader("Content-Type", "application/json");
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    var json = JSON.parse(xhr.responseText);
                }
            };
            xhr.send(JSON.stringify({data: this.metadata}));
        }else if(config.type == 'knn'){

        }else if(config.type == 'local'){
            let indexedDB = window.indexedDB || window.mozIndexedDB || window.webkitIndexedDB || window.msIndexedDB;
            if (!indexedDB) {
                throw new Error("Su navegador no soporta una versión estable de indexedDB. Por favor cambia la propiedad {type: 'local'} a {type: 'download'}");
                return false;
            }
            await this.model.save('indexeddb://model')
            localStorage.setItem('metadata', JSON.stringify(this.metadata, null, 4))
        }
        this.model.dispose();
    }
    /**
     *
     *
     * @param {*} file
     * @returns
     * @memberof TextClassifier
     */
    async fetch(file, cache = false){
        return new Promise((resolve, reject) => {
            if(cache){
                if(!localStorage.getItem('pixelity-' + file)){
                    var xmlHttp = new XMLHttpRequest();
                    xmlHttp.onreadystatechange = function() { 
                        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                            localStorage.setItem('pixelity-' + file, JSON.stringify({
                                update: moment().add(5, 'days'),
                                metadata: xmlHttp.responseText
                            }))
                            resolve(JSON.parse(xmlHttp.responseText));
                        }
                    }
                    xmlHttp.open("GET", file, true); // true for asynchronous 
                    xmlHttp.send(null);
                }else{
                    let m = JSON.parse(localStorage.getItem('pixelity-' + file))
                    if(moment().diff(moment(m.update)) < 0){
                        var xmlHttp = new XMLHttpRequest();
                        xmlHttp.onreadystatechange = function() { 
                            if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                                localStorage.setItem('pixelity-' + file, JSON.stringify({
                                    update: moment().add(5, 'days'),
                                    metadata: xmlHttp.responseText
                                }))
                                resolve(JSON.parse(xmlHttp.responseText));
                            }
                        }
                        xmlHttp.open("GET", file, true); // true for asynchronous 
                        xmlHttp.send(null);
                    }else{
                        resolve(m.metadata)
                    }
                }
            }else{
                var xmlHttp = new XMLHttpRequest();
                xmlHttp.onreadystatechange = function() { 
                    if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                        resolve(JSON.parse(xmlHttp.responseText));
                    }
                }
                xmlHttp.open("GET", file, true); // true for asynchronous 
                xmlHttp.send(null);
            }
        })
    }
    /**
     *
     *
     * @param {*} config
     * @returns
     * @memberof TextClassifier
     */
    async loadModel(config){
        return new Promise(async (resolve, reject) => {
            
            if(config.type == 'knn'){
                this.metadata = await this.fetch(config.metadataPath, false);
                resolve(this)
            }else{
                this.model = await tf.loadModel(tf.io.browserHTTPRequest(config.modelPath, {method: 'GET'}));
                this.metadata = await this.fetch(config.metadataPath, false);
                resolve(this)
            }
        })
    }
    /**
     *
     *
     * @param {*} input
     * @returns
     * @memberof TextClassifier
     */
    async prepareInput(input){
        let bag = Array(this.metadata.shape[0]).fill(0);
        let sentence = nlp(input).words().get();
        let words = []
        sentence.map((w, index) => words.push( nlp(w).stem(w.toLowerCase()).replace(/[?]/g, '')))
        this.metadata.words.map((word, index) => {
            words.map((w,i) => {
                if(w == word){
                    bag[index] = 1
                }
            })
        })
        return bag;     
    }
    /**
     *
     *
     * @param {*} intent
     * @memberof TextClassifier
     */
    stemPattern(intent){
        // loop through each sentense in our intents patterns
        intent.patterns.map((pattern, i) => {
            // tokenize each word in the sentence
            let w = nlp(pattern).words().get()
            // add to our word list
            this.words.extend(w);
            // add to documents in our corpus
            this.documents.push([w, intent.tag])
            // add to our classes list
            if(!this.classes.includes(intent.tag)){
                this.classes.push(intent.tag)
            }
        })
    }
}

export default TextClassifier;