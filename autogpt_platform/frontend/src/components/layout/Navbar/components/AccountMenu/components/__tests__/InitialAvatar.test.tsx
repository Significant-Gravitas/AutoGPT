import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { InitialAvatar } from "../InitialAvatar";

describe("InitialAvatar", () => {
  test("renders marble gradient fallback when no image is provided", () => {
    const { container } = render(<InitialAvatar name="abhimanyu" />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("uses trimmed name as fallback seed", () => {
    const { container } = render(<InitialAvatar name="   beth" />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("falls back to 'User' when name is missing", () => {
    const { container } = render(<InitialAvatar />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("falls back to 'User' when name is an empty string", () => {
    const { container } = render(<InitialAvatar name="" />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("merges className prop into the avatar root", () => {
    const { container } = render(
      <InitialAvatar name="ada" className="size-12" />,
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("size-12");
  });

  test("shows image when src is provided", async () => {
    render(<InitialAvatar name="ada" src="https://example.com/avatar.png" />);

    await waitFor(() => {
      expect(screen.getByRole("img", { name: "ada's avatar" })).toBeDefined();
    });
  });
});
