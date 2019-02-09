import * as tf from '@tensorflow/tfjs'
import Handsfree from 'handsfree'
import facesOfOz from './start_face'
import {TweenMax, Power2, TimelineLite} from "gsap/TweenMax";


Array.prototype.extend = function (other_array) {
    /* You should include a test to check whether other_array really is an array */
    other_array.forEach(function(v) {this.push(v)}, this);
}
/**
 *
 *
 * @class FaceControl
 */
class FaceControl{
    constructor(){
        this.model = new Handsfree();
        this.tweenTime = 500
        this.tween = {pitch: 0, yaw: 0}
        window.facesOfOz = facesOfOz;
    }

    start(config, cb){
        let self = this;
        this.model.use({
            name: 'youtube-360',
            
            /**
             * This is called on each webcam frame
             * @param {Array} faces An array of detected faces
             */
            onFrame: (faces) => {
                faces.map(face => {
                    TweenMax.to(this.tween, this.tweenTime / 1000, {
                        ease: 'Linear.easeNone',
                        pitch: -face.rotationX * 180 / Math.PI * 8,
                        yaw: -face.rotationY * 180 / Math.PI * 10,
                        roll: face.rotationZ * 180 / Math.PI * 2,
                        overwrite: true,
                        immediate: true
                    })
                })
                cb(faces, this.tween)
            }
        })
        this.model.start()

        if(config){
            if(config.showFace){
                const BoidsDebugger = this.model.use({
                    name: 'boids-debugger',
        
                    // The canvas context
                    canvas: [
                        // Boids
                        {
                            $: null,
                            ctx: null
                        },
                        // Orbs
                        {
                            $: null,
                            ctx: null
                        }
                    ],
        
                    // The element containing the canvases
                    $wrap: null,
        
                    // The offset to move boids to
                    offset: {
                        x: 0,
                        y: 0
                    },
        
                    // How fast to move by
                    rateOfChange: 0.01,
                    // How much extra time to wait before flying back
                    doYourOwnThingTimer: 3000,
                    // When a Boid is deleted, it's ID is added here for reuse
                    // Id's match a corresponding face landmark
                    freedIDs: [],
                    // The amount to modify acceleration
                    trackingSpeedMod: 0.9,
                    // Whether we're ready
                    isReady: false,
        
                    // The point boids return to
                    returnPoint: {
                        x: 0,
                        y: 0
                    },
        
                    // Defaulting to 64 since we have 64 face landmarks
                    maxBoids: 64,
                    // Collection of boid instances
                    boids: [],
        
                    orbSize: 7,
        
                    /**
                     * Setup the canvas and boids
                     */
                    onUse() {
                        // Add the wrapper
                        this.$wrap = document.createElement('div')
                        this.$wrap.classList.add('handsfree-boids-debugger-wrap')
                        this.$wrap.width = window.innerWidth
                        document.body.appendChild(this.$wrap)
        
                        // Add the canvas
                        for (let i = 1; i > -1; i--) {
                            this.canvas[i].$ = document.createElement('canvas')
                            this.canvas[i].ctx = this.canvas[i].$.getContext('2d')
                            this.$wrap.appendChild(this.canvas[i].$)
                        }
                        this.canvas[0].$.classList.add('handsfree-boids-debugger-primary-canvas')
                        this.canvas[1].$.classList.add('handsfree-boids-debugger-secondary-canvas')
                        document.body.addEventListener('mousedown', () => {
                            this.mouseDown = true
                            this.trackingSpeedMod = 0.009
                        })
                        document.body.addEventListener('mouseup', () => {
                            this.mouseDown = false
                        })
        
                        // Reponsive
                        setInterval(() => this.animateBoids(), 1000 / 29.9)
                        window.addEventListener('resize', () => this.onResize())
                        this.onResize()
        
                        // update point to where the mouse cursor is
                        document.onmousemove = e => this.returnPoint = {
                            x: e.pageX,
                            y: e.pageY
                        }
        
                        // Start boid loop
                        setTimeout(() => this.createInitialBoids(), 0)
                        setTimeout(() => {
                            this.rateOfChange = 0.98
                            this.doYourOwnThingTimer = 0
                        }, 3000)
                    },
        
                    /**
                     * Animates the boids
                     */
                    animateBoids() {
                        // Draw orbs
                        this.canvas[1].ctx.beginPath()
                        this.canvas[1].ctx.rect(0, 0, this.canvas[1].$.width, this.canvas[1].$.height)
                        this.canvas[1].ctx.fillStyle = 'rgba(18, 10, 34, 0.5)'
                        this.canvas[1].ctx.fill()
        
                        // Draw Boids
                        if (this.boids.length < this.maxBoids) {
                            const boid = new Boid()
                            boid.id = this.freedIDs.pop()
                            boid.color = self.model.getPointColor(boid.id)
                            this.boids.push(boid)
                        }
                        this.canvas[0].ctx.clearRect(0, 0, this.canvas[0].$.width, this.canvas[0].$.height)
                        this.canvas[0].ctx.globalAlpha = 0.1
        
                        // Remove any boids who's update method returns false,
                        // meaning it was out of bounds
                        this.boids = this.boids.filter(p => p.update())
                    },
        
                    /**
                    * Resizes an element to match the window size
                    */
                    resize($el) {
                        $el.width = window.innerWidth
                        $el.height = window.innerHeight
                    },
        
                    /**
                    * Resizes all canvas elements
                    */
                    onResize() {
                        this.resize(this.$wrap)
                        this.resize(this.canvas[0].$)
                        this.resize(this.canvas[1].$)
        
                        // Set the offset based on responsive mode so that it's always above the title
                        if (window.innerWidth < 960) {
                            this.offset.x = window.innerWidth / 2 - facesOfOz[0].translationX + 70
                        } else {
                            this.offset.x = window.innerWidth / 2 - facesOfOz[0].translationX + window.innerWidth / 4
                        }
                        this.offset.y = -facesOfOz[0].translationY / 4 + 40
                    },
        
                    onFrame() {
                        this.trackingSpeedMod += 0.0001
                        this.trackingSpeedMod = Math.min(this.trackingSpeedMod, 0.8)
                    },
        
                    /**
                    * Reset the faces to whatever the start scene is
                    */
                    onStop() {
                        self.model.faces = facesOfOz
                    },
        
                    onStart() {
                        this.trackingSpeedMod = 0.009
                        this.isReady = false
                        setTimeout(() => {
                            this.isReady = true
                        }, 1000)
                    },
        
                    /**
                    * Creates the 64 initial boids in a "face" position
                    */
                    createInitialBoids() {
                        for (let i = 0; i < 65; i++) {
                            const boid = new Boid()
                            boid.pos = {
                                x: facesOfOz[0].points[i].x + this.offset.x,
                                y: facesOfOz[0].points[i].y + this.offset.y
                            }
                            boid.color = self.model.getPointColor(i)
                            boid.id = i
                            this.boids.push(boid)
                        }
                    }
                })
        
                /**
                * Represents a single boid
                */
                const Boid = function() {
                    this.pos = {
                        x: Math.random() * BoidsDebugger.canvas[0].$.width * 0.8 + BoidsDebugger.canvas[0].$.width * 0.1,
                        y: Math.random() * BoidsDebugger.canvas[0].$.height * 0.8 + BoidsDebugger.canvas[0].$.height * 0.1
                    }
                    this.r = 1
                    this.speed = 6
                    this.step = Math.random() * 400
                    this.vx = Math.random() * this.speed / 4 - this.speed / 8
                    this.vy = Math.random() * this.speed / 4 - this.speed / 8
                    this.color = '#ff0'
                    this.history = []
        
                    this.update = function() {
                        //////////////////////////////////////
                        this.step++
                        this.step %= 400
                        if (this.history.length > 5) {
                            this.history.shift()
                        }
                        this.pos.x += this.vx
                        this.pos.y += this.vy
                        this.vx = this.vx * BoidsDebugger.rateOfChange + (Math.random() * this.speed * 2 - this.speed) * 0.12
                        this.vy = this.vy * BoidsDebugger.rateOfChange + (Math.random() * this.speed * 2 - this.speed) * 0.12
        
                        // Fly towards point
                        if (self.model.isTracking && BoidsDebugger.isReady) {
                            this.speed = 0.05
                            this.vx = this.vx * BoidsDebugger.trackingSpeedMod - (this.pos.x - self.model.faces[0].points[this.id].x - BoidsDebugger.offset.x) * 0.512
                            this.vy = this.vy * BoidsDebugger.trackingSpeedMod - (this.pos.y - self.model.faces[0].points[this.id].y) * 0.512
                            // BoidsDebugger.canvas[0].ctx.globalAlpha = 0.35
                            // Fly towards start point
                        } else if (!self.model.isTracking && BoidsDebugger.mouseDown) {
                            this.speed = 0.05
                            this.vx = this.vx * BoidsDebugger.trackingSpeedMod - (this.pos.x - facesOfOz[0].points[this.id].x - BoidsDebugger.offset.x - Math.random() * 1) * 0.512
                            this.vy = this.vy * BoidsDebugger.trackingSpeedMod - (this.pos.y - facesOfOz[0].points[this.id].y - BoidsDebugger.offset.y - Math.random() * 1) * 0.512
                            // Do their own thing
                        } else {
                            BoidsDebugger.canvas[0].ctx.globalAlpha = 0.1
                            this.speed = 6
                        }
        
                        //////////////////////////////////////
                        if (this.history.length > 4) {
                            // Boid Body
                            BoidsDebugger.canvas[1].ctx.beginPath()
                            BoidsDebugger.canvas[1].ctx.moveTo(this.pos.x, this.pos.y)
                            for (var i = this.history.length - 1; i >= 0; i--) {
                                BoidsDebugger.canvas[1].ctx.lineTo(this.history[i].x, this.history[i].y)
                            }
                            // BoidsDebugger.canvas[1].ctx.fillStyle = `hsla(${Math.sin( this.step / 300) * 70 + 70},${99}%,${50}%,1)`
                            // BoidsDebugger.canvas[1].ctx.strokeStyle = `hsla(${Math.sin( this.step / 300) * 70 + 70},${99}%,${50}%,0.5)`
                            BoidsDebugger.canvas[1].ctx.fillStyle = this.color
                            BoidsDebugger.canvas[1].ctx.strokeStyle = this.color
                            BoidsDebugger.canvas[1].ctx.fill()
                            BoidsDebugger.canvas[1].ctx.lineWidth = 2
                            BoidsDebugger.canvas[1].ctx.lineJoin = "round"
                            // BoidsDebugger.canvas[1].ctx.closePath()
                            BoidsDebugger.canvas[1].ctx.stroke()
        
                            // Boid orb
                            BoidsDebugger.canvas[0].ctx.beginPath()
                            BoidsDebugger.canvas[0].ctx.fillStyle = `rgba(250,250,250,0.05)`
                            BoidsDebugger.canvas[0].ctx.fillStyle = this.color
                            BoidsDebugger.canvas[0].ctx.arc(this.history[4].x, this.history[4].y, BoidsDebugger.orbSize, 0, 2 * Math.PI)
                            BoidsDebugger.canvas[0].ctx.fill()
                        }
        
                        // Delete the boid if it goes out of bounds
                        if (this.pos.x > BoidsDebugger.canvas[0].$.width || this.pos.x < 0 || this.pos.y > BoidsDebugger.canvas[0].$.height || this.pos.y < 0) {
                            BoidsDebugger.freedIDs.push(this.id)
                            delete this.pos
                            delete this.history
                            return false
                        }
        
                        // The boid is still in bounds, lets update its history
                        this.history.push({
                            x: this.pos.x,
                            y: this.pos.y
                        })
                        return true
                    }
                }
            }
        }
                
        
    }
}

export default FaceControl;