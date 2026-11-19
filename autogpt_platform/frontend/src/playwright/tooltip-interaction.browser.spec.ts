import { expect, test, type Locator, type Page } from "@playwright/test";

async function overlapControl(
  page: Page,
  target: Locator,
  wrapperEdge = false,
) {
  await page.locator("body").evaluate(async (body) => {
    await Promise.all(
      body
        .getAnimations({ subtree: true })
        .map((animation) => animation.finished.catch(() => undefined)),
    );
  });
  const box = await target.boundingBox();
  expect(box).not.toBeNull();
  const point = {
    x: box!.x + (wrapperEdge ? 2 : box!.width / 2),
    y: box!.y + box!.height / 2,
  };
  await page.getByTestId("underlying").evaluate((button, position) => {
    Object.assign(button.style, {
      position: "fixed",
      width: "20px",
      height: "20px",
      margin: "0",
      zIndex: "1",
    });
    const bounds = button.getBoundingClientRect();
    const style = getComputedStyle(button);
    button.style.left = `${parseFloat(style.left) + position.x - 10 - bounds.x}px`;
    button.style.top = `${parseFloat(style.top) + position.y - 10 - bounds.y}px`;
  }, point);
  return point;
}

for (const modal of [false, true]) {
  test.describe(modal ? "inside a modal" : "outside a modal", () => {
    for (const variant of ["ui", "atom", "atom-portal"]) {
      test(`${variant}: open tooltip content and wrapper pass clicks through`, async ({
        page,
      }) => {
        await page.goto(
          `/src/playwright/tooltip/index.html?variant=${variant}${modal ? "&modal" : ""}`,
        );
        const tooltip = page.locator(
          "[data-radix-popper-content-wrapper] > [data-state]",
        );
        await expect(tooltip).toBeVisible();
        const wrapper = tooltip.locator("..");
        await wrapper.evaluate((element) => {
          element.style.padding = "8px";
        });
        for (const [index, target] of [tooltip, wrapper].entries()) {
          const point = await overlapControl(page, target, index === 1);
          await expect(tooltip).toBeVisible();
          await page.mouse.click(point.x, point.y);
          await expect(
            page.getByRole("status", { name: "Underlying clicks" }),
          ).toHaveText(String(index + 1));
        }
      });
    }

    test("Markdown link stays reachable and receives a native click", async ({
      page,
      context,
    }) => {
      await context.route("**/tooltip-link-destination", (route) =>
        route.fulfill({ body: "Setup guide" }),
      );
      await page.goto(
        `/src/playwright/tooltip/index.html?variant=information${modal ? "&modal" : ""}`,
      );
      await page.getByRole("button", { name: "More information" }).hover();
      const tooltip = page.locator(
        "[data-radix-popper-content-wrapper] > [data-state]",
      );
      await expect(tooltip).toBeVisible();
      const link = tooltip
        .locator(":scope > p")
        .getByRole("link", { name: "setup guide" });
      const point = await overlapControl(page, link);
      await page.mouse.move(point.x, point.y, { steps: 12 });
      await expect(tooltip).toBeVisible();
      const popupPromise = page.waitForEvent("popup", { timeout: 5000 });
      await page.mouse.click(point.x, point.y);
      const popup = await popupPromise;
      try {
        await expect(popup).toHaveURL(/\/tooltip-link-destination$/);
        await expect(
          page.getByRole("status", { name: "Underlying clicks" }),
        ).toHaveText("0");
      } finally {
        await popup.close();
      }
    });
  });
}
