import { describe, expect, it } from "vitest";
import { nameCapturedPhotos } from "../helpers";

describe("nameCapturedPhotos", () => {
  it("renames the generic camera filename so photos stay distinguishable", () => {
    const [first, second] = nameCapturedPhotos([
      new File(["a"], "image.jpg", { type: "image/jpeg" }),
      new File(["b"], "image.jpg", { type: "image/jpeg" }),
    ]);

    expect(first.name).toMatch(/^photo-.*\.jpg$/);
    expect(second.name).toMatch(/^photo-.*-2\.jpg$/);
    expect(first.name).not.toBe(second.name);
  });

  it("leaves already-named files and non-images alone", () => {
    const named = new File(["a"], "receipt.png", { type: "image/png" });
    const doc = new File(["b"], "image.pdf", { type: "application/pdf" });

    const result = nameCapturedPhotos([named, doc]);

    expect(result[0]).toBe(named);
    expect(result[1]).toBe(doc);
  });
});
