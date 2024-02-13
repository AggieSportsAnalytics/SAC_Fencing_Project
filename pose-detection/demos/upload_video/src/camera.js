/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posedetection from '@tensorflow-models/pose-detection';

import * as params from './params';

import * as cv from "@techstark/opencv-js"
import DrawerElement from "../components/drawer.js";

export class Context {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output');
    this.source = document.getElementById('currentVID');
    this.ctx = this.canvas.getContext('2d');
    const stream = this.canvas.captureStream();
    const options = {mimeType: 'video/webm; codecs=vp9'};
    this.mediaRecorder = new MediaRecorder(stream, options);
    this.mediaRecorder.ondataavailable = this.handleDataAvailable;
  }

userAngleNumber = 0;
userAngles = {
    // _id: 0, 
    name: null, 
    elbow_left: null,
    hip_left: null,
    knee_left: null, 
    elbow_right: null,
    hip_right: null,
    knee_right: null 
};

  drawCtx() {
    this.ctx.drawImage(
        this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param poses A list of poses to render.
   */
  drawResults(poses) {
    for (const pose of poses) {
      this.drawResult(pose);
    }
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param pose A pose with keypoints to render.
   */
  drawResult(pose) {
    if (pose.keypoints != null) {
      this.drawSkeleton(pose.keypoints);
      this.drawKeypoints(pose.keypoints);
      this.displayAngles(pose.keypoints);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   */
  drawKeypoints(keypoints) {
    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);
    // this.ctx.fillStyle = 'Red';
    this.ctx.strokeStyle = 'Red';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    // leftArmPoints = [5, 7, 9]
    // this.ctx.fillStyle = 'Green';
    // for (const i of leftArmPoints) {
    //   this.drawKeypoint(keypoints[i]);
    // }
    // this.ctx.fillStyle = "Green";
    // for (const i of keypointInd.left) {
    //   this.drawKeypoint(keypoints[i]);
    // }

    // for (const i of keypointInd.middle) {
    //   this.drawKeypoint(keypoints[i]);
    // }

    // //this.ctx.fillStyle = 'Orange';
    // for (const i of keypointInd.right) {
    //   this.drawKeypoint(keypoints[i]);
    // }

    // this.ctx.fillStyle = 'Green';
    // for (const i of keypointInd.right) {
    //   this.drawKeypoint(keypoints[i]);
    // }

    this.ctx.fillStyle = "Red";
    let relevantPts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    for (const i of relevantPts) {
      this.drawKeypoint(keypoints[i]);
    }
  }

  drawKeypoint(keypoint) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      //circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      circle.arc(keypoint.x, keypoint.y, 10, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints.
   */
  drawSkeleton(keypoints) {
    this.ctx.fillStyle = 'Black';
    this.ctx.strokeStyle = 'Black';
    this.ctx.lineWidth = 5;//params.DEFAULT_LINE_WIDTH;

    let relevantPairs = [
                          [6, 8], [8, 10],
                          [5, 7], [7, 9],
                          [6, 12], [12, 14],
                          [5, 11], [11, 13],
                          [12, 14], [14, 16],
                          [11, 13], [13, 15]
                        ];

    relevantPairs.forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      // If score is null, just show the keypoint.
      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
        this.ctx.beginPath();
        this.ctx.moveTo(kp1.x, kp1.y);
        this.ctx.lineTo(kp2.x, kp2.y);
        this.ctx.stroke();
      }

    });

    // posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([
    //                                                                   i, j
    //                                                                 ]) => {
    //   const kp1 = keypoints[i];
    //   const kp2 = keypoints[j];

    //   console.log("\n");
    //   console.log(calculateAngle(keypoints[6], keypoints[8], keypoints[10]));
    //   console.log("\n");

    //   // If score is null, just show the keypoint.
    //   const score1 = kp1.score != null ? kp1.score : 1;
    //   const score2 = kp2.score != null ? kp2.score : 1;
    //   const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    //   if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
    //     this.ctx.beginPath();
    //     this.ctx.moveTo(kp1.x, kp1.y);
    //     this.ctx.lineTo(kp2.x, kp2.y);
    //     this.ctx.stroke();
    //   }
    // });
  }


  displayAngles(keypoints) {
    const kptriples = [
      [5, 7, 9], // left elbow
      [6, 8, 10], // right elbow
      [6, 12, 14], // right hip
      [5, 11, 13], // left hip
      [12, 14, 16], // right knee
      [11, 13, 15] // left knee
    ]
    // Slow down video according to elbow angle; we want to slow it down for
    // moments of extension
    // if (elbowAngle > 120) {
    //   this.video.playbackRate = 0.25;
    // } else if (elbowAngle > 90) {
    //   this.video.playbackRate = 0.5;
    // } else {
    //   this.video.playbackRate = 1;
    // }

    // Referenced following StackOverflow guide: https://stackoverflow.com/a/33138692
    const fontsize = 20;
    const fontface = 'roboto';
    this.ctx.font = "bold " + fontsize + 'px ' + fontface;
    const lineHeight = fontsize * 1.1;

    // For now, locking playback rate at 0.25
    this.video.playbackRate = 0.25;
    this.ctx.fillText("Playback Rate: " + this.video.playbackRate, this.video.videoWidth / 2, 20);

    kptriples.forEach((triple) => {
      const kp1 = keypoints[triple[0]];
      const kp2 = keypoints[triple[1]];
      const kp3 = keypoints[triple[2]];

      // Gets angle between points, limiting it to two decimal points then putting into string
      const elbowAngle = this.calculateAngle(kp1, kp2, kp3);      
      const angleText = "" + elbowAngle.toFixed(2);

      //FIXME: if statements
      if(triple[0] === 5 && triple[1] === 7 && triple[2] === 9) { this.userAngles.elbow_left = elbowAngle; }
      else if(triple[0] === 6 && triple[1] === 8 && triple[2] === 10) { this.userAngles.elbow_right = elbowAngle; }
      else if(triple[0] === 6 && triple[1] === 12 && triple[2] === 14) { this.userAngles.hip_left = elbowAngle; }
      else if(triple[0] === 5 && triple[1] === 11 && triple[2] === 13) { this.userAngles.hip_right = elbowAngle; }
      else if(triple[0] === 12 && triple[1] === 14 && triple[2] === 16) { this.userAngles.knee_left = elbowAngle; }
      else if(triple[0] === 11 && triple[1] === 13 && triple[2] === 15) { this.userAngles.knee_right = elbowAngle; }
      // console.log(`Triple: ${triple}`);
      // console.log(`Angle: ${elbowAngle}\nUpdated Angle: ${this.userAngles.elbow_left}`);

      // Place angle text just slightly off of the middle point (kp2)
      let x = kp2.x + 5;
      let y = kp2.y + 10;


      let textWidth = this.ctx.measureText(angleText).width;
      this.ctx.textAlign = 'left';
      this.ctx.textBaseline = 'top';
      this.ctx.fillStyle = 'Cyan';
      this.ctx.fillRect(x, y, textWidth, lineHeight);
      this.ctx.fillStyle = 'Black';
      this.ctx.fillText(angleText, x, y);
    });
  }

  calculateAngle(keypoint1, keypoint2, keypoint3) {
    //Start coordinates
    const x1 = keypoint1.x;
    const y1 = keypoint1.y;

    //Middle coordinates
    const x2 = keypoint2.x;
    const y2 = keypoint2.y;

    //End coordinates
    const x3 = keypoint3.x;
    const y3 = keypoint3.y;

    var angle = this.radiansToDegrees(Math.atan2(y3 - y2, x3 - x2) - Math.atan2(y1 - y2, x1 - x2));

    if (angle > 180) {
      angle = 360-angle;
    }

    return angle;
  }


  radiansToDegrees(radianAngles) {
    return Math.abs(radianAngles * (180/Math.PI));
  }

  start() {
    this.mediaRecorder.start();
  }

  stop() {
    this.mediaRecorder.stop();
  }

  inputToDB(pose) {
    if(pose != "Pause") {
      this.userAngles.name = pose;
      this.handleAnglesUpload(this.userAngles);
    }
  }

  async handleAnglesUpload(document) {
    try {
      const response = await fetch('http://localhost:3000/api/angles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: document.name, elbow_left: document.elbow_left, hip_left: document.hip_left, knee_left: document.knee_left, elbow_right: document.elbow_right, hip_right: document.hip_right, knee_right: document.knee_right }),
      });
      console.log("Handling upload");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json(); // Returns the added course with _id
    } catch (error) {
      console.error("Could not add the course to the database", error);
    }
  }

  getUserAngles() {
    return this.userAngles;
  }

  handleDataAvailable(event) {
    if (event.data.size > 0) {
      const recordedChunks = [event.data];

      // Download.
      const blob = new Blob(recordedChunks, {type: 'video/webm'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      document.body.appendChild(a);
      a.style = 'display: none';
      a.href = url;
      a.download = 'pose.webm';
      a.click();
      window.URL.revokeObjectURL(url);
    }
  }
}