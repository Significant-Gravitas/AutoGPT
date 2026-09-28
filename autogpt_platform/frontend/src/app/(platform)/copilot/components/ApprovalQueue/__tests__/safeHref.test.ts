import { expect, test } from "vitest";
import { safeHref } from "../helpers";

test.each([
  ["/library?folder=x", "/library?folder=x"],
  ["/team/exp-1", "/team/exp-1"],
  ["//evil.com", null],
  ["/\\evil.com", null],
  ["\\\\evil.com", null],
  ["https://evil.com", null],
  ["/", null],
  [null, null],
])("safeHref(%j) is %j", (href, expected) => {
  expect(safeHref(href)).toBe(expected);
});
