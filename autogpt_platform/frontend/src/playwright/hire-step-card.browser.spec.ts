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
