import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { MCPAuthSchemeField } from "../MCPAuthSchemeField";

describe("MCPAuthSchemeField", () => {
  it("renders a plain label with the given class by default", () => {
    render(
      <MCPAuthSchemeField
        value="bearer"
        onChange={vi.fn()}
        labelClassName="mt-1"
      />,
    );

    const label = screen.getByText("Authentication type");
    expect(label.tagName).toBe("LABEL");
    expect(label.getAttribute("data-slot")).toBeNull();
    expect(label.className).toBe("mt-1");
  });

  it("renders the label through Text when given a variant, still naming the select", () => {
    const onChange = vi.fn();
    render(
      <MCPAuthSchemeField
        value="bearer"
        onChange={onChange}
        labelVariant="body-medium-tight"
      />,
    );

    const label = screen.getByText("Authentication type");
    expect(label.tagName).toBe("LABEL");
    expect(label.getAttribute("data-slot")).toBe("text");
    expect(label.className).toContain("font-medium");
    expect(label.className).toContain("text-inherit");

    const select = screen.getByLabelText("Authentication type");
    fireEvent.change(select, { target: { value: "basic" } });
    expect(onChange).toHaveBeenCalledWith("basic");
  });
});
