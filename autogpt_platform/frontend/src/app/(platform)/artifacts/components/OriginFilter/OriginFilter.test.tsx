import { describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { OriginFilter } from "./OriginFilter";

describe("OriginFilter", () => {
  test("renders all three tabs", () => {
    render(<OriginFilter value="all" onChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /all/i })).toBeDefined();
    expect(screen.getByRole("tab", { name: /builder/i })).toBeDefined();
    expect(screen.getByRole("tab", { name: /autopilot/i })).toBeDefined();
  });

  test("marks the active tab as aria-selected", () => {
    render(<OriginFilter value="builder" onChange={() => {}} />);
    const builder = screen.getByRole("tab", { name: /builder/i });
    const all = screen.getByRole("tab", { name: /all/i });
    expect(builder.getAttribute("aria-selected")).toBe("true");
    expect(all.getAttribute("aria-selected")).toBe("false");
  });

  test("clicking a tab forwards the value to onChange", () => {
    const onChange = vi.fn();
    render(<OriginFilter value="all" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /autopilot/i }));
    expect(onChange).toHaveBeenCalledWith("autopilot");
  });

  test("clicking the already-active tab still fires onChange (caller decides)", () => {
    const onChange = vi.fn();
    render(<OriginFilter value="builder" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /builder/i }));
    expect(onChange).toHaveBeenCalledWith("builder");
  });
});
