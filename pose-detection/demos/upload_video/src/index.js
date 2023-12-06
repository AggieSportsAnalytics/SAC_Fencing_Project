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

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {setupStats} from './stats_panel';
import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setBackendAndEnvFlags} from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const statusElement = document.getElementById('status');

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, {modelType});
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
      camera.video,
      {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});

  endEstimatePosesStats();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }
}
// async function renderResult(format) {
//   // FPS only counts the time it takes to finish estimatePoses.
//   beginEstimatePosesStats();

//   const poses = await detector.estimatePoses(
//       format,
//       {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});

//   endEstimatePosesStats();

//   camera.drawCtx(format);

//   // The null check makes sure the UI is not in the middle of changing to a
//   // different model. If during model change, the result is from an old
//   // model, which shouldn't be rendered.
//   if (poses.length > 0 && !STATE.isModelChanged) {
//     camera.drawResults(poses, format);
//   }
// }

// document.getElementById("startCam").addEventListener("click", () => {
//   const video = document.getElementById("vid");

//   if (navigator.mediaDevices.getUserMedia) {
//     navigator.mediaDevices
//       .getUserMedia({ video: true })
//       .then((stream) => {
//         video.srcObject = stream;
//         renderResult(video);
//       })
//       .catch(function (error) {
//         console.log("Something went wrong!");
//       });
//   }
// });
document.getElementById("startCam").addEventListener("click", () => {
  const video = document.getElementById("vid");

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
        renderResult();
      })
      .catch(function (error) {
        console.log("Something went wrong!");
      });
  }
});

document.getElementById("dropdown").addEventListener(("change"), () => {
  showSelected();
});

function showSelected() {
  var dropdown = document.getElementById("dropdown");
  var selectedOption = document.getElementById("selectedOption");
  var startCam = document.getElementById("startCam");
  var stopCam = document.getElementById("stopCam");
  var upload = document.getElementById("upload");
  var run = document.getElementById("submit");

  // Get the selected option's text content
  var selectedText = dropdown.options[dropdown.selectedIndex].text;

  // Display the selected option
  if(selectedText != "Select the video type")
  {
    selectedOption.textContent = "Selected Option: " + selectedText;
  }

  if(selectedText == "Uploaded Video")
  {
    run.style.display = "inline-block";
    startCam.style.display = "none";
    stopCam.style.display = "none";
    upload.style.display = "inline-block";
  }
  else if(selectedText == "Live feed")
  {
    run.style.display = "inline-block";
    startCam.style.display = "inline-block";
    stopCam.style.display = "inline-block";
    upload.style.display = "none";
  }
  else 
  {
    selectedOption.textContent = "Selected Option: none";
    run.style.display = "none";
    startCam.style.display = "none";
    stopCam.style.display = "none";
    upload.style.display = "none";
  }
}

document.getElementById("stopCam").addEventListener("click", () => {
  const video = document.getElementById("vid");

  const stream = video.srcObject;

  if (stream) {
    const tracks = stream.getTracks();

    tracks.forEach(function (track) {
      track.stop();
    });

    video.srcObject = null;
  }
});

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.video.currentSrc);
  const file = event.target.files[0];
  camera.source.src = URL.createObjectURL(file);

  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;

  statusElement.innerHTML = 'Video is loaded.';
}

async function runFrame() {
  await checkGuiUpdate();
  if (video.paused) {
    // video has finished.
    camera.mediaRecorder.stop();
    camera.clearCtx();
    camera.video.style.visibility = 'visible';
    return;
  }
  await renderResult();
  rafId = requestAnimationFrame(runFrame);
}
// async function runFrame(format) {
//   await checkGuiUpdate();
//   if (video.paused) {
//     // video has finished.
//     camera.mediaRecorder.stop();
//     camera.clearCtx(format);
//     camera.video.style.visibility = 'visible';
//     return;
//   }
//   await renderResult(format);
//   rafId = requestAnimationFrame(runFrame);
// }


async function run() {
  statusElement.innerHTML = 'Warming up model.';

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
        tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimatePoses(
        warmUpTensor,
        {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  camera.video.style.visibility = 'hidden';
  video.pause();
  video.currentTime = 0;
  video.play();
  camera.mediaRecorder.start();

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  await runFrame();
}
// async function run() {
//   var dropdown = document.getElementById("dropdown");

//   // Get the selected option's text content
//   var selectedText = dropdown.options[dropdown.selectedIndex].text;
//   if(selectedText == "Live feed") {
//     format = document.getElementById('vid');
//   }
//   else {
//     format = document.getElementById('video');
//   }

//   statusElement.innerHTML = 'Warming up model.';

//   // Warming up pipeline.
//   const [runtime, $backend] = STATE.backend.split('-');

//   if (runtime === 'tfjs') {
//     const warmUpTensor =
//         tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
//     await detector.estimatePoses(
//         warmUpTensor,
//         {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
//     warmUpTensor.dispose();
//     statusElement.innerHTML = 'Model is warmed up.';
//   }

//   camera.video.style.visibility = 'hidden';
//   video.pause();
//   video.currentTime = 0;
//   video.play();
//   camera.mediaRecorder.start();

//   await new Promise((resolve) => {
//     camera.video.onseeked = () => {
//       resolve(video);
//     };
//   });

//   await runFrame(format);
// }

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  await tf.ready();
  detector = await createDetector();

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;
};

//Timer logic starts from here
let startTime = 0;
let elapsedTime = 0;
let currentTime = 0;
let countdownTime = 0;
//let started = false;
let paused = true;
let intervalId;
let mins = 0;
let secs = 0;
let milliseconds = 0;

let currentInstruction = 0;
let instructions = [
  "Perform an en guarde...",
  "Perform an advance...",
  "Perform a lunge...",
  "Peform a defensive stance..."
];

// startButton.addEventListener("click", () => {
//   if (paused && started) {
//     // Timer had been started before, simply restarting the timer
//       paused = false;
//       startTime = Date.now() - elapsedTime;
//       intervalId = setInterval(updateTime, 1);
//   } else {
//     // First time timer has been started
//     started = true;
//     paused = false;
//     countdownTime = 3;
//     timeDisplay.textContent = countdownTime;
//     intervalId = setInterval(() => {
//       //timeDisplay.textContent = countdownTime;
//       if (countdownTime > 1) {
//         // 1 should be the last number displayed
//         countdownTime--;
//         timeDisplay.textContent = countdownTime;
//       } else {
//         clearInterval(intervalId);
//         startTime = Date.now() - elapsedTime;
//         intervalId = setInterval(updateTime, 1);
//       }
//     }, 1000)
//   }
// });

startButton.addEventListener("click", () => {
  if (paused && currentInstruction < instructions.length) {
    paused = false;

    instructionDisplay.textContent = instructions[currentInstruction];
    currentInstruction++;

    countdownTime = 3;
    timeDisplay.textContent = countdownTime;
    intervalId = setInterval(() => {
      if (countdownTime > 1) {
        // 1 should be the last number displayed
        countdownTime--;
        timeDisplay.textContent = countdownTime;
      } else {
        // End of countdown, start actual stopwatch
        clearInterval(intervalId);
        startTime = Date.now() - elapsedTime;
        instructionDisplay.textContent += " GO!";
        intervalId = setInterval(updateTime, 1);
      }
    }, 1000)
  } else {
    instructionDisplay.textContent = "End of instructions";
  }
});

// pauseButton.addEventListener("click", () => {
//   if (paused == false) {
//     paused = true;
//     elapsedTime = Date.now() - startTime;
//     clearInterval(intervalId);
//   }
// });

pauseButton.addEventListener("click", () => {
  if (!paused) {
    paused = true;
    elapsedTime = Date.now() - startTime;
    startTime = 0;
    elapsedTime = 0;
    currentTime = 0;
    mins = 0;
    secs = 0;
    milliseconds = 0;
    clearInterval(intervalId);
  }
});

resetButton.addEventListener("click", () => {
  paused = true;
  clearInterval(intervalId);
  startTime = 0;
  elapsedTime = 0;
  currentTime = 0;
  mins = 0;
  secs = 0;
  milliseconds = 0;
  currentInstruction = 0;
  instructionDisplay.textContent = "Awaiting Instruction...";
  timeDisplay.textContent = "00:00:000";
});

function updateTime() {
  elapsedTime = Date.now() - startTime;

  mins = Math.floor((elapsedTime / (1000*60)) % 60);
  secs = Math.floor((elapsedTime / (1000)) % 60);
  milliseconds = Math.floor((elapsedTime) % 1000);

  mins = padZeroes(mins, 2);
  secs = padZeroes(secs, 2);
  milliseconds = padZeroes(milliseconds, 3);

  timeDisplay.textContent = `${mins}:${secs}:${milliseconds}`;

  function padZeroes(unit, desiredLength) {
    return unit.toString().padStart(desiredLength, "0")
    // return (("0") + unit).length > desiredLength ? unit : "0" + unit;
  }
}

app();
