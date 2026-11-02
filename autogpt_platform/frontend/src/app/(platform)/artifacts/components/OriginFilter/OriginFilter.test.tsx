import { describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { OriginFilter } from "./OriginFilter";

describe("OriginFilter", () => {
  test("renders all three tabs", () => {
    render(<OriginFilter value="all" onChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /all/i })).toBeDefined();
    expect(screen.getByRole("tab", { name: /uploaded/i })).toBeDefined();
    expect(screen.getByRole("tab", { name: /generated/i })).toBeDefined();
  });

  test("marks the active tab as aria-selected", () => {
    render(<OriginFilter value="uploaded" onChange={() => {}} />);
    const uploaded = screen.getByRole("tab", { name: /uploaded/i });
    const all = screen.getByRole("tab", { name: /all/i });
    expect(uploaded.getAttribute("aria-selected")).toBe("true");
    expect(all.getAttribute("aria-selected")).toBe("false");
  });

  test("clicking a tab forwards the value to onChange", () => {
    const onChange = vi.fn();
    render(<OriginFilter value="all" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /generated/i }));
    expect(onChange).toHaveBeenCalledWith("generated");
  });

  test("clicking the already-active tab still fires onChange (caller decides)", () => {
    const onChange = vi.fn();
    render(<OriginFilter value="uploaded" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /uploaded/i }));
    expect(onChange).toHaveBeenCalledWith("uploaded");
  });
});
