import * as tf from '@tensorflow/tfjs'

class LSTM {
    constructor(options) {
        if (options && 
            options.seqLength &&
            options.hiddenSize &&
            options.numLayers) {
            this.seqLength = options.seqLength;
            this.hiddenSize = options.hiddenSize;
            this.numLayers = options.numLayers;
            this.vocab = null;
            this.indexToVocab = null
            this.outputKeepProb = options.outputKeepProb;
        } else {
            throw new Error("Missing some needed parameters");
        }
    }
    async init(resolve, options) {
        const logger = options && options.logger ? options.logger : console.log;

        logger("setting up model...");

        let cells = [];
        for (let i = 0; i < this.numLayers; i++) {
            const cell = await tf.layers.lstmCell({
                units: this.hiddenSize
            });
            cells.push(cell);
        }

        const multiLstmCellLayer = await tf.layers.rnn({
            cell: cells,
            returnSequences: true,
            inputShape: [this.seqLength, this.vocab.size]
        });

        const dropoutLayer = await tf.layers.dropout({
            rate: this.outputKeepProb
        });

        const flattenLayer = tf.layers.flatten();

        const denseLayer = await tf.layers.dense({
            units: this.vocab.size,
            activation: 'softmax',
            useBias: true
        });

        const model = tf.sequential();
        model.add(multiLstmCellLayer);
        model.add(dropoutLayer);
        model.add(flattenLayer);
        model.add(denseLayer);

        logger("compiling...");

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: 'adam'
        });

        logger("done.");

        this.model = await model;
        resolve()
    }
    async train(options) {
        const logger = options && options.logger ? options.logger : console.log;
        const batchSize = options.batchSize;
        const epochs = options && options.epochs ? options.epochs : 1;
        this.model.summary()
        logger('Start training')
        for (let i = 0; i < epochs; i++) {
            logger("Epoch: " + i)
            const modelFit = await this.model.fit(this.trainIn, this.trainOut, {
                batchSize: batchSize,
                epochs: 1,
                callbacks: {
                    onEpochEnd: async (e, log) => {
                        console.log('Epoch: ' + e, 'Loss: ' + log.loss)
                        const prediction = await this.predict('Cuántas veces', 50)
                        console.log(prediction)
                    },
                    onTrainBegin: () =>{
                        console.log('start')
                    }
                }
            });
            logger("Loss after epoch " + (i + 1) + ": " + modelFit.history.loss[0]);
        }
    }

    async predict(primer, amnt) {
        primer = this.oneHotString(primer, this.vocab);
        let startIndex = primer.length - this.seqLength - 1;
        let output = tf.tensor(primer);
        for (let i = 0; i < amnt; i++) {
            let slicedVec = output.slice(i + startIndex, this.seqLength);
            slicedVec = slicedVec.reshape([1, slicedVec.shape[0], slicedVec.shape[1]]);
            let next = await this.model.predict(slicedVec, {
                batchSize: 1,
                epochs: 1
            });
            output = output.concat(next);
        }
        return this.decodeOutput(output, this.indexToVocab);
    }

    async compile(inputText, seqLength){
        return new Promise(async (resolve, reject) => {
            let [trainIn, trainOut, vocab, indexToVocab] = this.prepareData(inputText, seqLength);
            this.trainIn = trainIn
            this.trainOut = trainOut
            this.vocab = vocab
            this.indexToVocab = indexToVocab;
            await this.init(resolve)
        })
    }

    prepareData(text, seqLength) {
        let data = text.split("");
        let [vocab, indVocab] = this.getVocab(data);
        let dataX = [];
        let dataY = [];
        for (let i = 0; i < data.length - seqLength; i++) {
            let inSeq = data.slice(i, i + seqLength);
            let outSeq = data[i + seqLength];
            dataX.push(inSeq.map(x => this.oneHot(vocab.size, vocab.get(x))));
            dataY.push(this.oneHot(vocab.size, vocab.get(outSeq)));
        }
        return [tf.tensor(dataX), tf.tensor(dataY), vocab, indVocab];
    }
    
    oneHot(size, at) {
        let vector = [];
        for (let i = 0; i < size; i++) {
            if (at == i) {
                vector.push(1);
            } else {
                vector.push(0);
            }
        }
        return vector;
    }
    
    oneHotString(text, vocab) {
        let output = [];
        for (let i = 0; i < text.length; i++) {
            let onehot = this.oneHot(vocab.size, vocab.get(text.charAt(i)));
            output.push(onehot);
        }
        return output;
    }
    
    async decodeOutput(data, vocab){
        let output = [];
        for (let i = 0; i < data.shape[0]; i++) {
            let tensor = data.slice(i, 1);
            tensor = tensor.reshape([vocab.length])
            let index = tensor.argMax();
            index = await index.data();
            index = index[0];
            let letter = vocab[index];
            output.push(letter);
        }
        return output.join("");
    }
    
    getVocab(arr){
        //get letter mapped to amount of occurances
        let counts = new Map();
        for (let i of arr) {
            if (counts.has(i)) {
                const value = counts.get(i);
                counts.set(i, value + 1);
            } else {
                counts.set(i, 1);
            }
        }
        // here we are taking those occurances and turning it in
        // into a map from letter to how frequetly it appears relative to other letters
        let indVocab = [];
        let vocab = new Map(Array.from(counts).sort((a, b) => {
            return b[1] - a[1];
        }).map((value, i) => {
            indVocab.push(value[0]);
            return [value[0], i];
        }));
    
        return [vocab, indVocab];
    }
}

export default LSTM;