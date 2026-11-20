import { buildSafeWorkspaceDownloadHeaders } from "../app/api/proxy/[...path]/route.helpers";
import { expect, test } from "./coverage-fixture";

const PREVIEW_URL = "https://shared-files.test/preview.svg";
const ACTIVE_CONTENT_URL = "https://shared-files.test/payload.html";
const EXECUTION_MARKER_URL = "https://shared-files.test/executed";

const SVG_PREVIEW = `
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
    <rect width="20" height="10" fill="green" />
  </svg>
`;

const ACTIVE_CONTENT = `
  <script>
    window.name = "payload-executed";
    fetch("${EXECUTION_MARKER_URL}");
  </script>
`;

function safeHeaders(
  contentType: string,
  contentDisposition: string,
  body: string,
) {
  return buildSafeWorkspaceDownloadHeaders(
    contentType,
    contentDisposition,
    Buffer.byteLength(body),
  );
}

test.describe("shared workspace file delivery", () => {
  test("renders an attached SVG as an image subresource", async ({ page }) => {
    await page.route(PREVIEW_URL, async (route) => {
      await route.fulfill({
        status: 200,
        headers: safeHeaders(
          "image/svg+xml",
          'inline; filename="preview.svg"',
          SVG_PREVIEW,
        ),
        body: SVG_PREVIEW,
      });
    });

    await page.setContent(`<img id="preview" src="${PREVIEW_URL}" />`);

    const preview = page.locator("#preview");
    await expect
      .poll(
        async () =>
          await preview.evaluate(
            (image) => (image as HTMLImageElement).naturalWidth,
          ),
      )
      .toBe(20);
  });

  test("downloads active content without executing it", async ({ page }) => {
    let executionMarkerRequested = false;
    page.on("request", (request) => {
      if (request.url() === EXECUTION_MARKER_URL) {
        executionMarkerRequested = true;
      }
    });

    await page.route(ACTIVE_CONTENT_URL, async (route) => {
      await route.fulfill({
        status: 200,
        headers: safeHeaders(
          "text/html; charset=utf-8",
          'inline; filename="payload.html"',
          ACTIVE_CONTENT,
        ),
        body: ACTIVE_CONTENT,
      });
    });

    await page.goto("data:text/html,<script>window.name='safe-page'</script>");
    const downloadPromise = page.waitForEvent("download");
    await page.goto(ACTIVE_CONTENT_URL).catch(() => undefined);
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toBe("payload.html");
    expect(await page.evaluate(() => window.name)).toBe("safe-page");
    expect(executionMarkerRequested).toBe(false);
  });
});
