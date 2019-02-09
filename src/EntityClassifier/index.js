import * as tf from '@tensorflow/tfjs'
import nlp from 'lorca-nlp'
import shuffle from 'shuffle-array'
import moment from 'moment';
import { NerManager } from 'node-nlp' 

Array.prototype.extend = function (other_array) {
    /* You should include a test to check whether other_array really is an array */
    other_array.forEach(function(v) {this.push(v)}, this);
}
/**
 *
 *
 * @class EntityClassifier
 */
class EntityClassifier{
    constructor(){
        this.lang = 'es'
        this.manager = new NerManager({ threshold: 0.8 }) 
    }

    addEntityText(entity){
        this.manager.addNamedEntityText(entity.entity, entity.option, [this.lang], entity.patterns)
    }

    addEntity(entity){
        let ent = this.manager.addNamedEntity(entity.entity, entity.option)
        if(entity.between){
            ent.addBetweenCondition(this.lang, entity.between[0], entity.between[1])
        }
        if(entity.afterLast){
            ent.addAfterLastCondition(this.lang, entity.afterLast)
        }
    }

    async predict(input){
        return this.manager.findEntities(input,this.lang)
    }
}

export default EntityClassifier;