import html from "choo/html";
import Component from "choo/component";

import { css } from "@emotion/css";

import p5 from "p5";

const mainCss = css`
width: 100%;
height: 100vh;
overflow: hidden;

canvas {
  width: 100%;
  height: 100%;
}
`;

export default class extends Component {
  constructor(id, state, emit) {
    super(id)
    this.local = state.components[id] = {}
    this.state = state
    this.emit = emit
  }

  load(element) {
    const p = this.sketch();
    this.state.p5 = p;
    p.chooState = this.state;
    // BAD
    const polling = () => {
      if (p.canvas === undefined) {
        console.log("canvas not found, retrying");
        setInterval(polling, 100);
      }
      else {
        element.appendChild(p.canvas);
        p.resizeCanvas(element.clientWidth, element.clientHeight);
        p.parentElement = element;
        p.canvas.style = "";
      }
    }
    polling();
  }
  
  sketch() {
    const s = ( p ) => {
      p.setup = () => {
        const canvas = p.createCanvas(window.innerWidth, window.innerHeight);
      };

      p.draw = () => {
        p.clear();
        p.fill("crimson");
        p.text("test", 50,50);
      };
    };

    return new p5(s);
  }

  update(center) {
    return false
  }

  createElement({ width = window.innerWidth, height = window.innerHeight } = {}) {

    return html`<div class=${ mainCss }>
    </div>`
  }
}