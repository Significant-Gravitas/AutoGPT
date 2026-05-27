import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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
      <InitialAvatar name="ada" className="size-12" />,
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("size-12");
  });

  test("places image above the initial fallback when src is provided", async () => {
    const { container } = render(
      <InitialAvatar name="ada" src="https://example.com/avatar.png" />,
    );

    await waitFor(() => {
      expect(container.querySelector("img")).not.toBeNull();
    });

    const image = container.querySelector("img");
    expect(image?.className).toContain("z-10");
    expect(screen.getByText("A").className).toContain("z-0");
  });
});
