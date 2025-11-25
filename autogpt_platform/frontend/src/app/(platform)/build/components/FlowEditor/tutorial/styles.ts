// styles.ts
import { CSS_CLASSES } from "./constants";

export const injectTutorialStyles = () => {
  const style = document.createElement("style");
  style.id = "tutorial-styles";
  style.textContent = `
    .${CSS_CLASSES.DISABLE} {
      pointer-events: none;
      opacity: 0.5;
    }
    .${CSS_CLASSES.HIGHLIGHT} {
      background-color: #ffeb3b;
      border: 2px solid #fbc02d;
      transition: background-color 0.3s, border-color 0.3s;
    }
  `;

  // Prevent duplicate injection
  if (!document.getElementById("tutorial-styles")) {
    document.head.appendChild(style);
  }
};

export const removeTutorialStyles = () => {
  const style = document.getElementById("tutorial-styles");
  if (style) {
    style.remove();
  }
};
