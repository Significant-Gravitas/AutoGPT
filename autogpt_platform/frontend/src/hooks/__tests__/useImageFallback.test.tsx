import { describe, expect, test } from "vitest";
import { act, renderHook } from "@testing-library/react";

import { useImageFallback } from "../useImageFallback";

describe("useImageFallback", () => {
  test("shows the image while the URL loads", () => {
    const { result } = renderHook(() => useImageFallback("https://cdn/a.png"));
    expect(result.current.showImage).toBe(true);
  });

  test("hides the image once the URL fails", () => {
    const { result } = renderHook(() => useImageFallback("https://cdn/a.png"));
    act(() => result.current.handleImageError());
    expect(result.current.showImage).toBe(false);
  });

  test("treats an absent URL as no image", () => {
    expect(
      renderHook(() => useImageFallback(null)).result.current.showImage,
    ).toBe(false);
    expect(
      renderHook(() => useImageFallback("")).result.current.showImage,
    ).toBe(false);
  });

  test("clears a previous failure when the URL changes", () => {
    const { result, rerender } = renderHook(
      ({ src }) => useImageFallback(src),
      { initialProps: { src: "https://cdn/a.png" } },
    );
    act(() => result.current.handleImageError());
    expect(result.current.showImage).toBe(false);

    rerender({ src: "https://cdn/b.png" });
    expect(result.current.showImage).toBe(true);
  });
});
