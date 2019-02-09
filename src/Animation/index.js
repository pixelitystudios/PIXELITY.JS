class Anim{
    constructor(element, animation){
        this.anims = []
        this.canvas = document.getElementById(element);
        this.image = document.createElement('img')
        this.image.className = 'viewport-image animated'
        this.animation = animation
        this.sequences = {}
        this.stopAnim = false;
    }

    setAnims(anims){
        this.anims = anims;
        for(let anim of this.anims){
            if(anim.initial){
                this.changeImage(anim, this.animation, 0)
            }
            this.setFunction(anim.name);
        }
    }

    changeImage(anim, animation, time, cb){
        setTimeout(() => {
            this.image.className = 'viewport-image animated ' + animation;
            this.image.src = anim.image
            this.canvas.innerHTML = ''
            this.canvas.appendChild(this.image)
            
            setTimeout(() => {
                this.image.className = 'viewport-image animated ' + animation;
                if(cb){
                    cb()
                }
            }, 500)
        }, time)
    }

    getAnim(name){
        for(let anim of this.anims){
            if(anim.name == name){
                return anim;
            }
        }
    }

    processSequence(sequences, index, counter, repeat){
        if(!this.stopAnim){
            if(repeat == 'infinite'){
                let seqs = Object.keys(sequences)
                let name = seqs[index];
                if(name){
                    let time = sequences[name];
                    this[name](time, () => {
                        if(index < seqs.length - 1 ){
                            this.processSequence(sequences, index + 1, counter, repeat)
                        }else{
                            this.processSequence(sequences, 0, counter, repeat)
                        }
                    })
                }
            }else{
                if(counter <= repeat){
                    let seqs = Object.keys(sequences)
                    let name = seqs[index];
                    if(name){
                        let time = sequences[name];
                        this[name](time, () => {
                            if(index == 0){
                                counter = counter + 1;
                            }
                            this.processSequence(sequences, index + 1, counter, repeat)
                        })
                    }
                }
            }
        }
    }

    stop(){
        this.stopAnim = true;
    }

    program(name, sequences){
        this.sequences[name] = (repeat) => {
            this.processSequence(sequences, 0, 0, repeat)
        }
    }

    start(anim, repeat){
        this.stopAnim = false;
        this.sequences[anim](repeat)
    }

    setFunction(name){
        this[name] = (time, cb) => {
            let anim = this.getAnim(name);
            this.changeImage(anim, this.animation, time, cb)
        }
    }

}

export default Anim;