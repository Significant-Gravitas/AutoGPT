export function translateText(
  element: HTMLElement,
  original: string,
  translated: string,
) {
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.textContent !== original) continue;
    const replacement = document.createElement("font");
    replacement.textContent = translated;
    node.parentNode!.replaceChild(replacement, node);
    return replacement;
  }
  throw new Error(`Could not find text to translate: ${original}`);
}
