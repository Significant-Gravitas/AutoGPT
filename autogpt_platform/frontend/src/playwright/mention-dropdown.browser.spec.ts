import { expect, test, type Page } from "@playwright/test";

const MARGIN = 8;

async function openFixture(
  page: Page,
  viewport: { width: number; height: number },
  query: string,
) {
  await page.setViewportSize(viewport);
  await page.goto(`/src/playwright/mention-dropdown/index.html?${query}`);
  const list = page.getByRole("listbox", { name: "Mention suggestions" });
  await expect(list).toBeVisible();
  await waitForListToSettle(page);
  return list;
}

async function waitForListToSettle(page: Page) {
  await page.evaluate(async () => {
    const list = document.querySelector("[role=listbox]")!;
    const frame = () => new Promise(requestAnimationFrame);
    let last = "";
    for (let stable = 0; stable < 3; ) {
      await frame();
      const rect = JSON.stringify(list.getBoundingClientRect());
      stable = rect === last ? stable + 1 : 0;
      last = rect;
    }
  });
}

async function boxes(page: Page) {
  return page.evaluate(() => {
    const list = document.querySelector<HTMLElement>("[role=listbox]")!;
    const composer = document.querySelector("[data-testid=composer]")!;
    const toBox = (rect: DOMRect) => ({
      top: rect.top,
      bottom: rect.bottom,
      left: rect.left,
      right: rect.right,
    });
    return {
      list: toBox(list.getBoundingClientRect()),
      composer: toBox(composer.getBoundingClientRect()),
      scrolls: list.scrollHeight > list.clientHeight,
    };
  });
}

function expectInsideViewport(
  box: { top: number; bottom: number; left: number; right: number },
  viewport: { width: number; height: number },
) {
  expect(box.top).toBeGreaterThanOrEqual(MARGIN - 0.5);
  expect(box.bottom).toBeLessThanOrEqual(viewport.height - MARGIN + 0.5);
  expect(box.left).toBeGreaterThanOrEqual(0);
  expect(box.right).toBeLessThanOrEqual(viewport.width);
}

test("opens above a composer at the bottom of the screen", async ({ page }) => {
  const viewport = { width: 1440, height: 900 };
  await openFixture(page, viewport, "composer=bottom");
  const { list, composer } = await boxes(page);
  expect(list.bottom).toBeLessThanOrEqual(composer.top);
  expectInsideViewport(list, viewport);
});

test("opens below a composer at the top of the screen, first option clickable", async ({
  page,
}) => {
  const viewport = { width: 1440, height: 900 };
  await openFixture(page, viewport, "composer=top");
  const { list, composer } = await boxes(page);
  expect(list.top).toBeGreaterThanOrEqual(composer.bottom);
  expectInsideViewport(list, viewport);
  await page.getByRole("option", { name: "Report 1.md" }).click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-selected",
    "Report 1.md",
  );
});

test("shrinks a long list to the roomier side and scrolls when neither side fits", async ({
  page,
}) => {
  const viewport = { width: 1440, height: 420 };
  await openFixture(page, viewport, "composer=middle&files=18");
  const { list, scrolls } = await boxes(page);
  expectInsideViewport(list, viewport);
  expect(scrolls).toBe(true);
  const last = page.getByRole("option", { name: "Report 18.md" });
  await last.scrollIntoViewIfNeeded();
  await expect(last).toBeInViewport({ ratio: 1 });
});

test("stays inside the composer's width on a narrow screen", async ({
  page,
}) => {
  const viewport = { width: 390, height: 844 };
  await openFixture(page, viewport, "composer=bottom");
  const { list, composer } = await boxes(page);
  expectInsideViewport(list, viewport);
  expect(list.right).toBeLessThanOrEqual(composer.right + 0.5);
});

test("keeps the empty state on screen with the composer at the top", async ({
  page,
}) => {
  const viewport = { width: 1440, height: 900 };
  await openFixture(page, viewport, "composer=top&files=0");
  await expect(page.getByText("No matching files.")).toBeInViewport({
    ratio: 1,
  });
  expectInsideViewport((await boxes(page)).list, viewport);
});

test("hides with the composer when its scrolling container scrolls it away", async ({
  page,
}) => {
  const list = await openFixture(
    page,
    { width: 1440, height: 900 },
    "scroller&files=4",
  );
  await page.getByTestId("scroller").evaluate((el) => (el.scrollTop = 1000));
  await expect(list).toBeHidden();
  await page.getByTestId("scroller").evaluate((el) => (el.scrollTop = 0));
  await expect(list).toBeVisible();
});
