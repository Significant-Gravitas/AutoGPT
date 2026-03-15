import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS } from "../constants";

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

export const waitForInputValue = (
  selector: string,
  targetValue: string,
  timeout = 30000,
): Promise<void> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkInput = () => {
      const input = document.querySelector(selector) as HTMLInputElement;
      if (input) {
        const currentValue = input.value.toLowerCase().trim();
        const target = targetValue.toLowerCase().trim();

        if (currentValue.includes(target) || target.includes(currentValue)) {
          if (currentValue.length >= 4 || currentValue === target) {
            resolve();
            return;
          }
        }
      }

      if (Date.now() - startTime > timeout) {
        resolve();
      } else {
        setTimeout(checkInput, TUTORIAL_CONFIG.INPUT_CHECK_INTERVAL);
      }
    };
    checkInput();
  });
};

export const waitForSearchResult = (
  selector: string,
  timeout = 15000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkResult = () => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        resolve(null);
      } else {
        setTimeout(checkResult, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkResult();
  });
};

export const waitForAnyBlockCard = (
  timeout = 10000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkBlock = () => {
      const block = document.querySelector(
        TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX,
      );
      if (block) {
        resolve(block);
      } else if (Date.now() - startTime > timeout) {
        resolve(null);
      } else {
        setTimeout(checkBlock, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkBlock();
  });
};

export const focusElement = (selector: string): void => {
  const element = document.querySelector(selector) as HTMLElement;
  if (element) {
    element.focus();
  }
};

export const scrollIntoView = (selector: string): void => {
  const element = document.querySelector(selector);
  if (element) {
    element.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }
};

export const typeIntoInput = (selector: string, text: string) => {
  const input = document.querySelector(selector) as HTMLInputElement;
  if (input) {
    input.focus();
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
};

export const observeElement = (
  selector: string,
  callback: (element: Element) => void,
): MutationObserver => {
  const observer = new MutationObserver((mutations, obs) => {
    const element = document.querySelector(selector);
    if (element) {
      callback(element);
      obs.disconnect();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  const element = document.querySelector(selector);
  if (element) {
    callback(element);
    observer.disconnect();
  }

  return observer;
};

export const watchSearchInput = (
  targetValue: string,
  onMatch: () => void,
): (() => void) => {
  const input = document.querySelector(
    TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
  ) as HTMLInputElement;
  if (!input) return () => {};

  let hasMatched = false;

  const handler = () => {
    if (hasMatched) return;

    const currentValue = input.value.toLowerCase().trim();
    const target = targetValue.toLowerCase().trim();

    if (currentValue.length >= 4 && target.startsWith(currentValue)) {
      hasMatched = true;
      onMatch();
    }
  };

  input.addEventListener("input", handler);

  return () => {
    input.removeEventListener("input", handler);
  };
};
