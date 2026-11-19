import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import AppVerifiedPage from "../page";

describe("AppVerifiedPage", () => {
  test("tells the person to go back to the app", async () => {
    render(<AppVerifiedPage />);

    expect(
      await screen.findByRole("heading", { name: "Your email is verified" }),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "Open the app" })).toHaveProperty(
      "href",
      "autogpt-app://",
    );
  });
});
