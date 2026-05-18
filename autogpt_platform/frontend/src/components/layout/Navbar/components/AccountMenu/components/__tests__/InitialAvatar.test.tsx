import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { InitialAvatar } from "../InitialAvatar";

describe("InitialAvatar", () => {
  test("renders uppercase first character of name", () => {
    render(<InitialAvatar name="abhimanyu" />);
    expect(screen.getByText("A")).toBeDefined();
  });

  test("trims whitespace before extracting initial", () => {
    render(<InitialAvatar name="   beth" />);
    expect(screen.getByText("B")).toBeDefined();
  });

  test("falls back to 'U' when name is missing", () => {
    render(<InitialAvatar />);
    expect(screen.getByText("U")).toBeDefined();
  });

  test("falls back to 'U' when name is an empty string", () => {
    render(<InitialAvatar name="" />);
    expect(screen.getByText("U")).toBeDefined();
  });

  test("merges className prop into the avatar root", () => {
    const { container } = render(
      <InitialAvatar name="ada" className="h-12 w-12" />,
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("h-12");
    expect(root.className).toContain("w-12");
  });
});
