import p5 from "p5";

import P5Element from './p5.js';

export default class extends P5Element {
  constructor(id, state, emit) {
    super(id, state, emit)
  }

  sketch() {
    const s = ( p ) => {
      p.preload = () => {
      };

      p.setup = () => {
        const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
        p.pixelDensity(1);
      };

      p.draw = () => {
        // if (p.videoElement === undefined) {
        //   p.videoElement = new p5.Element(p.chooState.videoElement);
        //   console.log(p.videoElement)
        // }
        // p.clear();
        // if (p.videoElement) {
        //   p.image(p.videoElement, 0, 0, p.width, p.height);
        // }
        // else {
        //   p.background("crimson")
        // }
        
        p.clear();
        let flow = p.chooState.flow;
        if (flow === undefined) {
          return;
        }
        p.noStroke();
        p.fill(0);
        if (p.chooState.showCamera == true) {
          p.drawingContext.drawImage(p.chooState.videoElement, 0, 0, flow.cols, flow.rows);
        }
        else {
          p.rect(0, 0, flow.cols, flow.rows);
        }
        // p.text(flow.data32F[0], 10, 30);
        
        p.stroke("lime");
        for (let i = 0; i < flow.rows; i+=10) {
          for (let j = 0; j < flow.cols; j+=10) {
            p.line(j,i,j+flow.data32F[(j+i*flow.cols) * 2 + 0],i+flow.data32F[(j+i*flow.cols) * 2 + 1])
          }
        }
        p.fill(0);
        p.text(p.chooState.count, 10, 10);
        p.text(p.chooState.cvError, 10, 30);

      };
      
      p.windowResized = () => {
        p.resizeCanvas(p.parentElement.clientWidth, p.parentElement.clientHeight);
      }
    };

    return new p5(s);
  }
}