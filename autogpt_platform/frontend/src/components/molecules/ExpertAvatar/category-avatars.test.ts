import { expect, test } from "vitest";
import { resolveCategoryAvatarUrl } from "./helpers";

test("a category variant uses the same expert's matching artwork", () => {
  expect(
    resolveCategoryAvatarUrl("/experts/clay/v3/quinn.png", "finance"),
  ).toBe("/experts/clay/v4/quinn-finance.png");
  expect(
    resolveCategoryAvatarUrl("/experts/clay/v3/quinn.png", "research"),
  ).toBe("/experts/clay/v3/quinn.png");
  expect(
    resolveCategoryAvatarUrl("/experts/clay/v3/quinn.png", "support"),
  ).toBe("/experts/clay/v3/quinn.png");
});

test("uploads and generated URLs keep their saved appearance", () => {
  for (const url of [
    "https://cdn.test/my-avatar.png",
    "/api/store/media/custom.png",
    "/experts/clay/v1/finance.png",
  ]) {
    expect(resolveCategoryAvatarUrl(url, "marketing")).toBe(url);
  }
});
