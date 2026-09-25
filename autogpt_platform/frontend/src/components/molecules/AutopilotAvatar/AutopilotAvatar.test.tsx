import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { AutopilotAvatar } from "./AutopilotAvatar";

test("switches between packaged and transparent Otto at the requested size", () => {
  const { rerender } = render(<AutopilotAvatar size={80} />);
  expect(screen.getByRole("img").getAttribute("src")).toBe(
    "/autogpt-characters/v1.1/otto/neutral/96.webp",
  );

  rerender(<AutopilotAvatar size={160} transparent />);
  const image = screen.getByRole("img", {
    name: "Otto, your personal Head of AI",
  });
  expect(decodeURIComponent(image.getAttribute("src") ?? "")).toContain(
    "/experts/transparent/otto.webp",
  );
  expect(image.getAttribute("width")).toBe("160");
  expect(image.getAttribute("height")).toBe("160");
  expect(image.getAttribute("sizes")).toBe("160px");

  rerender(<AutopilotAvatar />);
  expect(screen.getByRole("img").getAttribute("src")).toBe(
    "/autogpt-characters/v1.1/otto/neutral/24.webp",
  );
});
