import { describe, expect, it } from "vitest";

import {
  getSidebarItemVariants,
  sidebarContainerVariants,
} from "../animations";

describe("sidebarContainerVariants", () => {
  it("staggers its children on show", () => {
    expect(sidebarContainerVariants.hidden).toEqual({});
    expect(sidebarContainerVariants.show).toMatchObject({
      transition: { delayChildren: 0.15, staggerChildren: 0.06 },
    });
  });
});

describe("getSidebarItemVariants", () => {
  it("returns an opacity-only fade when reduced motion is requested", () => {
    const variants = getSidebarItemVariants(true);
    expect(variants.hidden).toEqual({ opacity: 0 });
    expect(variants.show).toEqual({
      opacity: 1,
      transition: { duration: 0.2 },
    });
  });

  it("returns the blur+rise variant when motion is allowed", () => {
    const variants = getSidebarItemVariants(false);
    expect(variants.hidden).toMatchObject({
      opacity: 0,
      y: 8,
      filter: "blur(4px)",
    });
    expect(variants.show).toMatchObject({
      opacity: 1,
      y: 0,
      filter: "blur(0px)",
    });
  });
});
