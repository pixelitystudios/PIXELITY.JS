import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs'
import Transform from './Transform'
import Joints from './Joints';

class Posenet{
    constructor(config){
        this.config = {
            imageScaleFactor: 0.5,
            outputStride: 16,
            flipHorizontal: false
        }
        this.joints = new Joints()
        this.transform = new Transform(this.joints)
        this.model;
    }

    async loadModel(){
        this.model = await posenet.load()
    }
    

    async getPose(imageElement){
        const pose = await this.model.estimateSinglePose(imageElement, this.config.imageScaleFactor, this.config.flipHorizontal, this.config.outputStride)
        this.transform.updateKeypoints(pose.keypoints, 0.5)
        const head = this.transform.head()
        const rightShoulderAngle = this.transform.rotateJoint('leftShoulder', 'rightShoulder','rightElbow');
        const rightArmAngle = this.transform.rotateJoint('rightShoulder', 'rightElbow', 'rightWrist');
        const leftShoulderAngle = this.transform.rotateJoint('rightShoulder', 'leftShoulder', 'leftElbow');
        const lefArmAngle = this.transform.rotateJoint('leftShoulder', 'leftElbow', 'leftWrist');
        console.log(rightShoulderAngle, rightArmAngle, leftShoulderAngle, lefArmAngle)
        return pose;
    }
}


export default Posenet;