import { describe, expect, test } from "vitest";
import { clearedAnswer } from "./flowItems";

describe("clearedAnswer", () => {
  test("drops a restored custom role when reopening the category beat", () => {
    expect(clearedAnswer("category")).toStrictEqual({
      category: null,
      color: null,
      legacyRole: undefined,
    });
  });
});
