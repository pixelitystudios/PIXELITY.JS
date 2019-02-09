import AudioDecoder from '../Utils/AudioDecoder'

class Synth{
    constructor(){

    }

    async extract(audio){
        const decoder = new AudioDecoder(audio);
        return await decoder.decode()
    }
}

export default Synth;