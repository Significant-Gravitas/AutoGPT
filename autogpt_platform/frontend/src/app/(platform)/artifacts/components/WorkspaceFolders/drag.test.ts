import { describe, expect, test } from "vitest";

import { createFileDragImage, FILE_DRAG_MIME } from "./drag";

describe("createFileDragImage", () => {
  test("renders an off-screen chip containing the file name", () => {
    const el = createFileDragImage("report.pdf");
    expect(el.textContent).toContain("report.pdf");
    // Positioned off-screen so it never flashes on the page.
    expect(el.style.position).toBe("absolute");
  });

  test("escapes HTML special characters in the file name", () => {
    const el = createFileDragImage('<img src=x onerror="alert(1)">.txt');
    expect(el.innerHTML).not.toContain("<img");
    expect(el.innerHTML).toContain("&lt;img");
  });
});

describe("FILE_DRAG_MIME", () => {
  test("is a stable custom MIME type", () => {
    expect(FILE_DRAG_MIME).toBe("application/workspace-file-id");
  });
});
