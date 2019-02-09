import Emitter from '../Utils/event-emitter'
import EntityClassifier from '../EntityClassifier'
import LenguageProcessor from '../LenguageProcessor'
import * as annyang from '../Utils/annyang'
import math from 'mathjs'

class BotFramework{
    constructor(model){
        this.config = {
            name: 'PixelityBot',
            confidence: 0.51
        }
        this.model = model;
        this.entitiesModel = new EntityClassifier()
        this.isForm = false;
        this.context = null;
        this.rulesForm = {
            reservations: [
                {
                    name: 'people',
                    answer: '¿Cuantas personas?',
                    type: 'number',
                    search: {
                        type: 'after',
                        regex: ['amigos','personas','persona','colegas','compañeros','compañeras','amigas','primos','primas','hermanos','hermanas','amigo','amiga','colega','compañero','hermano','hermana']
                    }
                },
                {
                    name: 'date',
                    answer: '¿Que día?',
                    type: 'date'
                },
                {
                    name: 'phone',
                    answer: '¿Cúal es tu télefono?',
                    type: 'phones'
                }
            ]
        }
        this.commands = {}
        this.Form = {
            params: {},
            currentRule: 0,
            lastResponse: ''
        }
        this.observable = new Emitter();
        this.customResponse = ''
        this.internalCommands = {
            '*input': (input) => {
                this.observable.emit('recognition:end', input)
                this.processInput(input)
            }
        }
        annyang.init(this.internalCommands)
        annyang.addCommands(this.internalCommands)
        annyang.setLanguage('es-MX')
    }

    setContext(context){
        this.client.context = context;
        this.model.setContext(context)
    }

    getContext(){
        return this.model.getContext()
    }

    setCommands(commands){
        this.commands = commands;
        annyang.removeCommands();
        annyang.addCommands(commands)
        annyang.addCommands(this.internalCommands)
    }

    startSpeechRecognition(config){
        annyang.start(config)
    }

    stopSpeechRecognition(){
        annyang.abort()
    }

    setCommandDebug(debug){
        annyang.debug(debug)
    }

    setRules(rules){
        this.rulesForm = rules;
    }

    addMessage(message){
        let response = {
            from: 'bot',
            input: message,
            prediction: {accuracy: 0, intent: null},
            timestamp: new Date()
        }
        this.client.conversation.push(response)
        this.observable.emit('last:message', response)
        this.observable.emit('simple:response', response)
        this.saveSession(this.client)
        this.observable.emit('subscribe:client', this.client)
    }

    setResponse(input, res, prepend){
        let inputMessage = {
            from: 'user',
            input: input,
            timestamp: new Date()
        }
        this.client.conversation.push(inputMessage)
        this.observable.emit('last:message', inputMessage)
        let response = {
            from: 'bot',
            input: prepend + res,
            prediction: {accuracy: 0, intent: null},
            timestamp: new Date()
        }
        this.client.conversation.push(response)
        this.observable.emit('last:message', response)
        this.observable.emit('simple:response', response)
        this.saveSession(this.client)
        this.observable.emit('subscribe:client', this.client)
    }

    math(expression){
        try {
            return math.eval(expression);
        } catch (error) {
            return 'No puedo formular tu expresión'
        }
    }

    addRule(key, value){
        this.rulesForm[key] = value
    }

    setName(name){
        this.config.name = name;
    }

    setConfidence(confidence){
        this.config.confidence = confidence;
    }

    async fetch(url){
        return new Promise((resolve, reject) => {
            var xmlHttp = new XMLHttpRequest();
            xmlHttp.onreadystatechange = function() { 
                if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                    resolve(JSON.parse(xmlHttp.responseText));
                }
            }
            xmlHttp.open("GET", url, true); // true for asynchronous 
            xmlHttp.send(null);
        })
    }

    async findParams(types, input){
        return new LenguageProcessor(types, input)
    }

    async processInput(input){
        let prediction = await this.model.predict(input);
        let inputMessage = {
            from: 'user',
            input: input,
            timestamp: new Date()
        }
        this.client.conversation.push(inputMessage)
        this.observable.emit('last:message', inputMessage)
        let entities = []
        if(this.entitiesModel){
            entities = await this.entitiesModel.predict(input);
        }
        let response = {
            from: 'bot',
            input: (prediction.intent) ?  this.preProcessResponse(prediction.intent.responses[Math.floor(Math.random() * prediction.intent.responses.length)]) : this.customResponse,
            prediction: prediction,
            entities: entities,
            timestamp: new Date()
        }
        this.client.conversation.push(response)
        this.observable.emit('last:message', response)
        this.observable.emit('simple:response', response)
        this.saveSession(this.client)
        if(prediction.intent){
            this.observable.emit(prediction.intent.extras.action, {
                intent: prediction.intent,
                findParams: async (types) => {
                    return await this.findParams(types, input)
                }
            })
        }
        this.observable.emit('subscribe:client', this.client)
    }

    async createForm(rule, response, lastResponse){
        let rules = this.rulesForm[rule];
        if(response){
            let params = await response.findParams(rules);
            this.Form.params = params.parameters
        }
        this.isForm = true;
        this.rules = rules;
        this.Form.lastResponse = lastResponse
        this.Form.currentRule = 0;
        this.createFormResponse()
    }

    closeForm(){
        this.isForm = false;
        let res = {
            from: 'bot',
            input: this.Form.lastResponse,
            prediction: null,
            timestamp: new Date()
        }
        this.client.conversation.push(res)
        this.observable.emit('last:message', res)
        this.observable.emit('close:form', res, this.Form.params)
        this.saveSession(this.client)
    }

    createFormResponse(){
        let res = {
            from: 'bot',
            input: this.rules[this.Form.currentRule].answer,
            prediction: null,
            form: {
                rule: this.rules[this.Form.currentRule],
                value: (this.Form.params[this.rules[this.Form.currentRule].name] !== undefined) ? this.Form.params[this.rules[this.Form.currentRule].name] : null
            },
            timestamp: new Date()
        }
        this.client.conversation.push(res)
        this.observable.emit('last:message', res)
        this.observable.emit('form:response', res)
        this.saveSession(this.client)
    }

    validateRule(type, input){
        let isValid = true;
        let value = null
        if(type == 'number'){
            if(isNaN(input)){
                isValid = false;
            }
        }
        if(type == 'date'){
            let date = moment(input);
            let dateParse = DateModule.es.parseDate(input);
            if(!dateParse){
                if(!date.isValid()){
                    isValid = false;
                }
            }else{
                value = moment(dateParse).format('DD-MM-YYYY')
            }
        }
        if(type == 'emails'){
            var re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
            if(!re.test(String(input).toLowerCase())){
                isValid = false;
            }
        }
        if(type == 'phones'){
            var phoneno = /^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$/;
            if(!input.match(phoneno))
            {
                isValid = false;
            }
        }
        if(type == 'urls'){
            var pattern = new RegExp('^(https?:\/\/)?'+ // protocol
            '((([a-z\d]([a-z\d-]*[a-z\d])*)\.)+[a-z]{2,}|'+ // domain name
            '((\d{1,3}\.){3}\d{1,3}))'+ // OR ip (v4) address
            '(\:\d+)?(\/[-a-z\d%_.~+]*)*'+ // port and path
            '(\?[;&a-z\d%_.~+=-]*)?'+ // query string
            '(\#[-a-z\d_]*)?$','i'); // fragment locater
            if(!pattern.test(input)) {
                isValid = false;
            }
        }
        return {
            value: value,
            isValid: isValid
        };
    }

    on(event, listener){
        this.observable.on(event, listener);
    }

    preProcessResponse(response){
        const parsers = [
            {
                regex: '{client.name}',
                value: (this.client.information.name) ? this.client.information.name.split(' ')[0] : ''
            }
        ]
        for(let parser of parsers){
            response = response.replace(new RegExp(parser.regex, 'g'), parser.value)
        }
        return response;
    }

    async sendMessage(input){
        if(this.client){
            if(this.isForm){
                let check = this.validateRule(this.rules[this.Form.currentRule].type, input)
                if(check.isValid){
                    if(check.value){
                        this.Form.params[this.rules[this.Form.currentRule].name] = check.value;
                    }else{
                        this.Form.params[this.rules[this.Form.currentRule].name] = input;
                    }
                    this.Form.currentRule = this.Form.currentRule + 1;
        
                    if(this.Form.currentRule > this.rules.length - 1){
                        this.closeForm()
                    }else{
                        this.createFormResponse()
                    }
                }else{
                    this.createFormResponse()
                }
            }else{
                annyang.trigger(input)
            }
        }else{
            throw new Error("You have not assigned a client to the bot");
        }
    }

    setCustomResponse(message){
        this.customResponse = message;
    }

    getSession(){
        return JSON.parse(localStorage.getItem('pxcontextsession'))
    }

    saveSession(session){
        localStorage.setItem('pxcontextsession', JSON.stringify(session))
    }

    setClientInfo(info){
        this.client.information = info;
    }

    subscribe(client){
        this.client = {
            token: Math.random().toString(36).substr(2, 9),
            information: client,
            conversation: [],
            context: null,
            confidence: 0,
            sentiment: 0
        };
        this.saveSession(this.client)
        this.observable.emit('subscribe:client', this.client)
    }
}

export default BotFramework;