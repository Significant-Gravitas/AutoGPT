import { TUTORIAL_CONFIG } from "../constants";

export const waitForElement = (
  selector: string,
  timeout = 10000,
): Promise<Element> => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const checkElement = () => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        reject(new Error(`Element ${selector} not found within ${timeout}ms`));
      } else {
        setTimeout(checkElement, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkElement();
  });
};

export const focusElement = (selector: string): void => {
  const element = document.querySelector(selector) as HTMLElement;
  if (element) {
    element.focus();
  }
};
