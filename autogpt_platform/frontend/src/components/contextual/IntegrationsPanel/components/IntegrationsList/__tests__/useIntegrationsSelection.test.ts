import { act, renderHook } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { useIntegrationsSelection } from "../useIntegrationsSelection";

describe("useIntegrationsSelection", () => {
  test("starts empty with no items selected", () => {
    const { result } = renderHook(() => useIntegrationsSelection(["a", "b"]));
    expect(result.current.selectedIds).toEqual([]);
    expect(result.current.selectedCount).toBe(0);
    expect(result.current.allSelected).toBe(false);
    expect(result.current.isSelected("a")).toBe(false);
  });

  test("toggle adds, then removes the same id", () => {
    const { result } = renderHook(() => useIntegrationsSelection(["a", "b"]));

    act(() => result.current.toggle("a"));
    expect(result.current.isSelected("a")).toBe(true);
    expect(result.current.selectedIds).toEqual(["a"]);
    expect(result.current.selectedCount).toBe(1);
    expect(result.current.allSelected).toBe(false);

    act(() => result.current.toggle("a"));
    expect(result.current.isSelected("a")).toBe(false);
    expect(result.current.selectedIds).toEqual([]);
  });

  test("selectAll selects every id and reports allSelected true", () => {
    const { result } = renderHook(() => useIntegrationsSelection(["a", "b"]));
    act(() => result.current.selectAll());
    expect(result.current.selectedCount).toBe(2);
    expect(result.current.allSelected).toBe(true);
    expect(result.current.isSelected("a")).toBe(true);
    expect(result.current.isSelected("b")).toBe(true);
  });

  test("clear empties the selection", () => {
    const { result } = renderHook(() => useIntegrationsSelection(["a", "b"]));
    act(() => result.current.selectAll());
    act(() => result.current.clear());
    expect(result.current.selectedCount).toBe(0);
    expect(result.current.allSelected).toBe(false);
  });

  test("drops ids that disappear from the available list when allIds shrinks", () => {
    const { result, rerender } = renderHook(
      ({ ids }) => useIntegrationsSelection(ids),
      { initialProps: { ids: ["a", "b", "c"] } },
    );
    act(() => result.current.toggle("a"));
    act(() => result.current.toggle("b"));
    expect(result.current.selectedCount).toBe(2);

    // 'b' goes away — selection should follow.
    rerender({ ids: ["a", "c"] });
    expect(result.current.isSelected("a")).toBe(true);
    expect(result.current.isSelected("b")).toBe(false);
    expect(result.current.selectedCount).toBe(1);
  });
});
