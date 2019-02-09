import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs'
import TextClassifier from './TextClassifier'
import BotFramework from './BotFramework'
import LenguageProcessor from './LenguageProcessor'
import FaceID from './FaceID'
import LSTM from './LSTMTextGenerator'
import AudioDecoder from './Utils/AudioDecoder'
import AudioEncoder from './Utils/AudioEncoder'
import ConcatenateBlobs from './Utils/ConcatenateBlobs'
import Anim from './Animation'
import Synth from './Synth'
import ImageClassification from './ImageClassification'
import EntityClassifier from './EntityClassifier'
import FeatureExtraction from './FeatureExtraction'
import ImageClassifier from './ImageClassifier'
import Posenet from './Posenet'
import FaceControl from './FaceControl'

module.exports = {
    TextClassifier,
    BotFramework,
    LenguageProcessor,
    FaceID,
    AudioDecoder,
    AudioEncoder,
    ConcatenateBlobs,
    EntityClassifier,
    LSTM,
    Anim,
    Synth,
    ImageClassification,
    ImageClassifier,
    FeatureExtraction,
    Posenet,
    FaceControl,
    tf,
}
