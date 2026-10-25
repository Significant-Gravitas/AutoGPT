import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { ButtonGroup, ButtonGroupText } from "../button-group";

describe("ButtonGroupText", () => {
  test("renders its copy through Text inside a group", () => {
    render(
      <ButtonGroup>
        <ButtonGroupText className="border-none">2 of 3</ButtonGroupText>
      </ButtonGroup>,
    );
    expect(screen.getByRole("group")).toBeDefined();
    const text = screen.getByText("2 of 3");
    expect(text.tagName).toBe("DIV");
    expect(text.getAttribute("data-slot")).toBe("text");
    expect(text.className).toContain("border-none");
    expect(text.className).toContain("font-medium");
  });
});
