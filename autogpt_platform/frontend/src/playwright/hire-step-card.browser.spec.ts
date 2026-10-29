import { expect, test } from "@playwright/test";

test("a left-out expert's tick takes a mouse click to put them back in", async ({
  page,
}) => {
  await page.goto("/src/playwright/hire-step-card/index.html");
  const tick = page.getByRole("button", { name: "Include Maria" });
  await expect(tick).toHaveAttribute("aria-pressed", "true");

  for (const pressed of ["false", "true"]) {
    await page.evaluate(() =>
      Promise.all(document.getAnimations().map((a) => a.finished)),
    );
    const box = await tick.boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.click(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await expect(tick).toHaveAttribute("aria-pressed", pressed);
  }
});

test("the tick stays under a z-10 overlay of the list, like its edge fades", async ({
  page,
}) => {
  await page.goto("/src/playwright/hire-step-card/index.html?covered");
  const tick = await page
    .getByRole("button", { name: "Include Maria" })
    .boundingBox();
  const overlay = await page.getByTestId("overlay").boundingBox();
  expect(tick).not.toBeNull();
  expect(overlay).not.toBeNull();
  const size = { width: tick!.width, height: tick!.height };
  const overTick = await page.screenshot({ clip: { ...tick!, ...size } });
  const overBlank = await page.screenshot({
    clip: { x: overlay!.x + 2, y: overlay!.y + 2, ...size },
  });
  expect(overTick.equals(overBlank)).toBe(true);
});
