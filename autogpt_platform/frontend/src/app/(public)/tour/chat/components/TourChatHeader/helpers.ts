export function buildTourShareUrl() {
  return `${window.location.origin}/tour/chat?utm_source=share`;
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // In-app browsers (Instagram, Google app) often block the async
    // clipboard API — fall back to the legacy execCommand path.
    return copyViaHiddenTextarea(text);
  }
}

function copyViaHiddenTextarea(text: string): boolean {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    textarea.remove();
  }
}
