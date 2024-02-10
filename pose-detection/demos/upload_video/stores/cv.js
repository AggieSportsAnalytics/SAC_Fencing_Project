import html from "choo/html";
import cv from "@techstark/opencv-js";
import { css } from "@emotion/css";

const videoCss = css`
  position: absolute;
  top: 100px;
  left: 0;
  width: 10px;
  height: 10px;
  opacity: 0;
`;


export default function (state, emitter) {
  let video = html`<video id="webcam" autoplay muted playsinline width="640" height="480" class=${ videoCss }></video>`;
  document.body.appendChild(video);
  state.videoElement = video;
  state.showCamera = true;
  let streaming = false;

  emitter.on("start capture", () => {
    // Check if webcam access is supported.
    function getUserMediaSupported() {
      return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }
    if (getUserMediaSupported()) {
    } else {
      console.warn("getUserMedia() is not supported by your browser");
      return;
    }
    
    // getUsermedia parameters to force video but not audio.
    const constraints = {
      video: true,
      width: 640,
      height: 480,
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      video.srcObject = stream;
      video.addEventListener("loadeddata", runOpticalFlow);
      streaming = true;
    });

    function runOpticalFlow() {
      let cap = new cv.VideoCapture(video);

      // take first frame and find corners in it
      let oldFrame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      cap.read(oldFrame);
      let oldGray = new cv.Mat();
      cv.cvtColor(oldFrame, oldGray, cv.COLOR_RGB2GRAY);

      let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      let frameGray = new cv.Mat();
      let flow = new cv.Mat();
      
      let count = 0;

      const FPS = 30;
      function processVideo() {
        try {
          state.cvError = "";
          if (!streaming) {
            // clean and stop.
            frame.delete();
            oldGray.delete();
            return;
          }
          let begin = Date.now();

          // start processing.
          cap.read(frame);
          // console.log("reading");
          cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);

          // calculate optical flow
          cv.calcOpticalFlowFarneback(oldGray, frameGray, flow, 0.5, 3, 20, 3, 5, 1.2, 0);

 // console.log(flow.cols, flow.rows, flow.data32F[0]);
          state.flow = flow;
          
          frameGray.copyTo(oldGray);

          let delay = 1000 / FPS - (Date.now() - begin);
          setTimeout(processVideo, Math.max(0, delay));
          state.count = count++;
        } catch (err) {
          console.log(err);
          state.cvError = err;
        }
      }

      // schedule the first one.
      setTimeout(processVideo, 0);
    }
  });
}