import { describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { ExpertFilter } from "./ExpertFilter";

const EXPERTS = [
  { id: "expert-a", name: "Nova", avatarUrl: "https://cdn.test/nova.png" },
  { id: "expert-b", name: "Kai", avatarUrl: null },
];

describe("ExpertFilter", () => {
  test("renders Everyone plus one tab per expert", () => {
    render(<ExpertFilter experts={EXPERTS} value={null} onChange={() => {}} />);
    expect(screen.getByRole("tab", { name: "Everyone" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Nova" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Kai" })).toBeDefined();
  });

  test("marks the selected expert as active", () => {
    render(
      <ExpertFilter experts={EXPERTS} value="expert-b" onChange={() => {}} />,
    );
    expect(
      screen.getByRole("tab", { name: "Kai" }).getAttribute("aria-selected"),
    ).toBe("true");
    expect(
      screen
        .getByRole("tab", { name: "Everyone" })
        .getAttribute("aria-selected"),
    ).toBe("false");
  });

  test("forwards the expert id, and null for Everyone", () => {
    const onChange = vi.fn();
    render(<ExpertFilter experts={EXPERTS} value={null} onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: "Nova" }));
    expect(onChange).toHaveBeenCalledWith("expert-a");
    fireEvent.click(screen.getByRole("tab", { name: "Everyone" }));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  test("renders nothing when no expert is hired and none is selected", () => {
    render(<ExpertFilter experts={[]} value={null} onChange={() => {}} />);
    expect(screen.queryByRole("tablist")).toBeNull();
  });

  test("keeps Everyone so a filter on a fired expert can be cleared", () => {
    const onChange = vi.fn();
    render(
      <ExpertFilter experts={[]} value="expert-gone" onChange={onChange} />,
    );
    fireEvent.click(screen.getByRole("tab", { name: "Everyone" }));
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
