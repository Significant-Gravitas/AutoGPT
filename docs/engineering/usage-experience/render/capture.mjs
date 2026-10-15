import fs from "node:fs/promises";
import path from "node:path";
import http from "node:http";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import cases from "./cases.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../../..");
const require = createRequire(
  path.join(repo, "autogpt_platform/frontend/package.json"),
);
const { chromium } = require("@playwright/test");
const bundle = process.env.USAGE_RENDER_OUTPUT || "/tmp/autogpt-usage-render";
const output = path.resolve(
  process.env.USAGE_SCREENSHOTS_OUTPUT || "./usage-component-screenshots",
);
const clock = "2026-10-07T12:00:00.000Z";
const widths = (process.env.USAGE_SCREENSHOT_WIDTHS || "1280,1000,390,320")
  .split(",")
  .map(Number);
const selected = process.env.USAGE_SCREENSHOT_CASES?.split(",");
const scenarios = selected
  ? cases.filter(({ id }) => selected.includes(id))
  : cases;
const manifest = {
  clock,
  locale: "en-US",
  timeZone: "UTC",
  reducedMotion: "reduce",
  componentSource: await fs
    .readFile(path.join(bundle, "source-manifest.json"), "utf8")
    .then(JSON.parse),
  captures: [],
  errors: [],
  cleanup: {},
};
let browser;
let server;
try {
  await fs.mkdir(output, { recursive: true });
  const mime = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".woff2": "font/woff2",
    ".html": "text/html",
  };
  server = http.createServer(async (req, res) => {
    try {
      const urlPath = new URL(req.url, "http://localhost").pathname;
      if (urlPath === "/favicon.ico") {
        res.writeHead(204).end();
        return;
      }
      const file = path.join(bundle, urlPath === "/" ? "index.html" : urlPath);
      if (!file.startsWith(path.resolve(bundle) + path.sep)) {
        res.writeHead(403).end();
        return;
      }
      res.setHeader(
        "Content-Type",
        mime[path.extname(file)] || "application/octet-stream",
      );
      res.end(await fs.readFile(file));
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.CHROMIUM_EXECUTABLE }
      : {}),
    args: ["--disable-dev-shm-usage"],
  });
  manifest.browserVersion = browser.version();
  for (const width of widths) {
    const context = await browser.newContext({
      viewport: { width, height: 1100 },
      deviceScaleFactor: 1,
      locale: "en-US",
      timezoneId: "UTC",
      reducedMotion: "reduce",
    });
    try {
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
      });
      await page.clock.setFixedTime(new Date(clock));
      for (const scenario of scenarios) {
        const { id, experience, screen } = scenario;
        await page.setViewportSize({ width, height: 1100 });
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        try {
          await page.waitForFunction(() => window.__implementationPreview);
        } catch (error) {
          console.error(JSON.stringify({ id, width, pageErrors: errors }));
          throw error;
        }
        await page.evaluate(
          ({ experience, screen }) =>
            window.__implementationPreview.show(experience, screen),
          { experience, screen },
        );
        await page.evaluate(() => document.fonts.ready);
        await page.waitForTimeout(350);
        const modal = !["billing", "chat", "notice"].includes(screen);
        const target = modal
          ? page.getByRole("dialog").first()
          : page.locator(
              screen === "billing"
                ? '[data-capture="billing"] > div'
                : `[data-capture="${screen}"]`,
            );
        try {
          await target.waitFor({ state: "visible" });
        } catch (error) {
          console.error(
            JSON.stringify({
              id,
              width,
              pageErrors: errors,
              body: await page.locator("body").innerText(),
            }),
          );
          throw error;
        }
        // A tall viewport reveals the whole existing drawer without altering its
        // styles or clipping its payment terms/controls. The manifest records it.
        if (modal) {
          for (let attempt = 0; attempt < 8; attempt++) {
            const clipped = await target.evaluate((element) =>
              [...element.querySelectorAll("*")]
                .concat(element)
                .reduce((extra, node) => {
                  const style = getComputedStyle(node);
                  return /auto|scroll/.test(style.overflowY)
                    ? Math.max(extra, node.scrollHeight - node.clientHeight)
                    : extra;
                }, 0),
            );
            if (clipped <= 1) break;
            await page.setViewportSize({
              width,
              height: page.viewportSize().height + Math.ceil(clipped) + 100,
            });
            await page.waitForTimeout(80);
          }
        }
        if (errors.length) throw new Error(`${id}: ${errors.join("; ")}`);
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth + 1,
        );
        if (overflow)
          throw new Error(`Horizontal document overflow: ${id} at ${width}px`);
        const clipped = await target.evaluate((element) =>
          [...element.querySelectorAll("*")]
            .concat(element)
            .filter((node) => {
              const style = getComputedStyle(node);
              return (
                /auto|scroll/.test(style.overflowY) &&
                node.scrollHeight > node.clientHeight + 1
              );
            })
            .map((node) => ({
              tag: node.tagName,
              hiddenHeight: node.scrollHeight - node.clientHeight,
            })),
        );
        if (clipped.length)
          throw new Error(
            `Scrollable content omitted in ${id}: ${JSON.stringify(clipped)}`,
          );
        const links = [];
        for (const control of await target
          .locator("button,a[href],summary")
          .all()) {
          if (!(await control.isVisible()) || (await control.isDisabled()))
            continue;
          const label = (
            (await control.getAttribute("aria-label")) ||
            (await control.textContent()) ||
            ""
          ).trim();
          await control.click({ trial: true, timeout: 5000 });
          links.push(label);
        }
        await page.mouse.move(0, 0);
        // Verify real persisted text and editable composition while send is paused.
        if (screen === "chat") {
          const editor = target.getByRole("textbox");
          if (!(await editor.isEditable()))
            throw new Error(`Draft cannot be edited: ${id}`);
          if (!(await target.locator('button[type="submit"]').isDisabled()))
            throw new Error(`Paused draft can send: ${id}`);
          if (
            !((await editor.textContent()) || "").includes(
              "Turn these findings into a launch plan.",
            )
          )
            throw new Error(`Saved draft missing: ${id}`);
        }
        const file = `${id}-${width}.png`;
        let componentBounds = await target.boundingBox();
        if (screen === "chat") {
          // The production notice floats above the composer. Include both
          // bounds instead of changing that layout or clipping the notice.
          const notice = await target
            .getByRole("status", { name: "Usage notice" })
            .boundingBox();
          const top = Math.min(componentBounds.y, notice.y);
          componentBounds = {
            ...componentBounds,
            y: top,
            height: componentBounds.y + componentBounds.height - top,
          };
          await page.screenshot({
            path: path.join(output, file),
            clip: componentBounds,
            animations: "disabled",
          });
        } else {
          await target.screenshot({
            path: path.join(output, file),
            animations: "disabled",
          });
        }
        const entry = {
          ...scenario,
          file,
          viewport: page.viewportSize(),
          componentBounds,
          content: await target.innerText(),
          reachableControls: links,
          horizontalOverflow: false,
          hiddenScrollableContent: false,
          sha256: crypto
            .createHash("sha256")
            .update(await fs.readFile(path.join(output, file)))
            .digest("hex"),
        };
        if (screen === "ready") {
          await target
            .getByRole("button", { name: "Continue where you left off" })
            .click();
          await target.waitFor({ state: "hidden" });
          const composer = page.locator('[data-capture="chat"]');
          const editor = composer.getByRole("textbox");
          if (
            !(await editor.isEditable()) ||
            (await composer.locator('button[type="submit"]').isDisabled())
          )
            throw new Error(`Ready draft cannot send: ${id}`);
          entry.afterDismissal = { draftEditable: true, sendEnabled: true };
        }
        manifest.captures.push(entry);
        console.log(
          JSON.stringify({
            id,
            width,
            height: entry.componentBounds.height,
            file,
          }),
        );
      }
    } finally {
      await context.close();
    }
  }
} catch (error) {
  manifest.errors.push(error.stack || String(error));
  console.error(error);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server?.listening) await new Promise((resolve) => server.close(resolve));
  manifest.cleanup = {
    browserClosed: !browser?.isConnected(),
    serverClosed: !server?.listening,
  };
  await fs.writeFile(
    path.join(output, "capture-manifest.json"),
    JSON.stringify(manifest, null, 2),
  );
  console.log(
    JSON.stringify({
      output,
      captures: manifest.captures.length,
      errors: manifest.errors.length,
      ...manifest.cleanup,
    }),
  );
}
