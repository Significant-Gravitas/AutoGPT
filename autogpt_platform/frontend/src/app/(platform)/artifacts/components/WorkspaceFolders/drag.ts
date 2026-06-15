/** DataTransfer MIME type used when dragging a file card onto a folder. */
export const FILE_DRAG_MIME = "application/workspace-file-id";

const FILE_GLYPH = `<svg width="20" height="20" viewBox="0 0 256 256" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M213.66,82.34l-56-56A8,8,0,0,0,152,24H56A16,16,0,0,0,40,40V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V88A8,8,0,0,0,213.66,82.34ZM160,51.31,188.69,80H160ZM200,216H56V40h88V88a8,8,0,0,0,8,8h48V216Z"></path></svg>`;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Builds an off-screen "file illustration" element used as the drag image when
 * dragging a file card, so the cursor carries a compact file chip instead of a
 * snapshot of the whole card. Caller is responsible for removing it on dragend.
 */
export function createFileDragImage(fileName: string): HTMLElement {
  const el = document.createElement("div");
  el.style.cssText = [
    "position:absolute",
    "top:-1000px",
    "left:-1000px",
    "display:inline-flex",
    "align-items:center",
    "gap:8px",
    "padding:8px 12px",
    "border-radius:12px",
    "border:1px solid #e4e4e7",
    "background:#ffffff",
    "color:#71717a",
    "font:500 13px ui-sans-serif,system-ui,sans-serif",
    "box-shadow:0 4px 12px rgba(0,0,0,0.08)",
  ].join(";");
  el.innerHTML = `${FILE_GLYPH}<span style="color:#18181b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px;">${escapeHtml(
    fileName,
  )}</span>`;
  return el;
}
