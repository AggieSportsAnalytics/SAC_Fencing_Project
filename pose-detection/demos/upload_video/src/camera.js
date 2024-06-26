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
      this.displayAngle(pose.keypoints);
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
    let relevantPts = [6, 8, 10];
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

    let rightLimbPairs = [[6, 8], [8, 10]]
    rightLimbPairs.forEach(([i, j]) => {
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


  displayAngle(keypoints) {
    const kp1 = keypoints[6];
    const kp2 = keypoints[8];
    const kp3 = keypoints[10];
    // Gets angle between points, limiting it to two decimal points then putting into string
    const elbowAngle = this.calculateAngle(kp1, kp2, kp3);
    const angleText = "" + elbowAngle.toFixed(2);

    // Slow down video according to elbow angle; we want to slow it down for
    // moments of extension
    if (elbowAngle > 120) {
      this.video.playbackRate = 0.25;
    } else if (elbowAngle > 90) {
      this.video.playbackRate = 0.5;
    } else {
      this.video.playbackRate = 1;
    }


    // Place angle text just slightly off of the middle point (kp2)
    const x = kp2.x + 5;
    const y = kp2.y + 10;

    // Referenced following StackOverflow guide: https://stackoverflow.com/a/33138692
    const fontsize = 20;
    const fontface = 'roboto';
    this.ctx.font = "bold " + fontsize + 'px ' + fontface;

    this.ctx.fillText("Playback Rate: " + this.video.playbackRate, this.video.videoWidth / 2, 20)

    const lineHeight = fontsize * 1.1;
    const textWidth = this.ctx.measureText(angleText).width;
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
    this.ctx.fillStyle = 'Cyan';
    this.ctx.fillRect(x, y, textWidth, lineHeight);
    this.ctx.fillStyle = 'Black';
    this.ctx.fillText(angleText, x, y);

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