import nlp from 'lorca-nlp'
import DateModule from 'chrono-node'
import moment from 'moment'
import 'moment/locale/es'
import math from 'mathjs'

class LenguageProcessor{
    constructor(types, input){
        moment.locale('es')
        this.processing = nlp
        this.dateProcessor = DateModule;
        this.input = input;
        this.types = types;
        this.intent = {
            queryText: input,
            parameters: {}
        }
        if(types && input){
            return this.process()
        }
    }

    process(){
        for(let type of this.types){
            if(type.type == 'date'){
                let dateParse = this.dateProcessor.es.parseDate(this.input);
                let date = this.dateProcessor.es.parse(this.input)[0];
                this.intent.parameters.date = moment(dateParse).format('DD-MM-YYYY')
            }
            if(type.type == 'emails'){
                this.intent.parameters[type.name] = this.extractEmails(this.input)
            }
            if(type.type == 'phones'){
                this.intent.parameters[type.name] = this.extractPhoneNumbers(this.input)
            }
            if(type.type == 'urls'){
                this.intent.parameters[type.name] = this.extractURL(this.input)
            }
            if(type.type == 'number'){
                if(type.search){
                    if(type.search.type == 'after'){
                        for(let p of type.search.regex){
                            let regex = new RegExp(p);
                            if(regex.test(this.input)){
                                let sp = this.input.split(p);
                                let match = sp[0].match(/\d+/g);
                                if(match){
                                    let numbers = match.map(Number);
                                    let final = numbers[numbers.length - 1];
                                    this.intent.parameters[type.name] = final
                                }
                            }
                        }
                    }
                    if(type.search.type == 'before'){
                        for(let p of type.search.regex){
                            let regex = new RegExp(p);
                            if(regex.test(this.input)){
                                let sp = this.input.split(p);
                                let match = sp[1].match(/\d+/g);
                                if(match){
                                    let numbers = match.map(Number);
                                    let final = numbers[numbers.length - 1];
                                    this.intent.parameters[type.name] = final
                                }
                            }
                        }
                    }
                    if(type.search.type == 'all'){
                        let numbers = this.input.match(/\d+/g).map(Number);
                        this.intent.parameters[type.name] = numbers[0]
                    }
                }
            }
        }
        return this.intent;
    }

    extractPeople(input, after){
        let peopleSearch = after
        
        for(let p of peopleSearch){
            let regex = new RegExp(p);
            if(regex.test(input)){
                let sp = input.split(p);
                let numbers = sp[0].match(/\d+/g).map(Number);
                let final = numbers[numbers.length - 1];
                return {
                    text: final + ' ' + p,
                    value: final,
                    search: p
                }
            }
        }
    }

    guid(){
        function s4() {
          return Math.floor((1 + Math.random()) * 0x10000)
            .toString(16)
            .substring(1);
        }
        return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
    }

    extractURL(text){
        let urls = text.match(/(?:(?:https?|ftp):\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,}))\.?)(?::\d{2,5})?(?:[\/?#]\S*)?/gi);
        let e = []
        if(urls){
            for(let url of urls){
                let split = url.split('://');
                return url;
            }
        }
        return null
    }

    extractEmails(text){
        let emails = text.match(/(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])/gi);
        let e = []
        if(emails){
            for(let email of emails){
                let split = email.split('@');
                return email
            }
        }
        return null
    }

    extractPhoneNumbers(text){
        let phones = text.match(/(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?/gi);
        let e = []
        if(phones){
            for(let phone of phones){
                return phone;
            }
        }
        return null;
    }

}

export default LenguageProcessor;