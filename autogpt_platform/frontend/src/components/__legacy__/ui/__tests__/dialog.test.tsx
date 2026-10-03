import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../dialog";

describe("legacy DialogContent", () => {
  test("injects an sr-only Description when none is provided", () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Untitled dialog</DialogTitle>
          </DialogHeader>
          <p>Body</p>
        </DialogContent>
      </Dialog>,
    );

    const dialog = screen.getByRole("dialog");
    const descriptionId = dialog.getAttribute("aria-describedby");
    expect(descriptionId).toBeTruthy();
    const description = document.getElementById(descriptionId ?? "");
    expect(description?.textContent).toBe("Dialog");
    expect(description?.classList.contains("sr-only")).toBe(true);
  });

  test("keeps an explicit DialogDescription without injecting a fallback", () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm</DialogTitle>
            <DialogDescription>This action cannot be undone.</DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>,
    );

    const dialog = screen.getByRole("dialog");
    const descriptionId = dialog.getAttribute("aria-describedby");
    expect(descriptionId).toBeTruthy();
    const description = document.getElementById(descriptionId ?? "");
    expect(description?.textContent).toBe("This action cannot be undone.");
    expect(screen.queryByText("Dialog", { selector: "p.sr-only" })).toBeNull();
  });

  test("respects a caller-provided aria-describedby without injecting a fallback", () => {
    render(
      <Dialog open>
        <DialogContent aria-describedby="custom-description">
          <DialogHeader>
            <DialogTitle>Export</DialogTitle>
          </DialogHeader>
          <p id="custom-description">Downloads a JSON file.</p>
        </DialogContent>
      </Dialog>,
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog.getAttribute("aria-describedby")).toBe("custom-description");
    expect(screen.queryByText("Dialog", { selector: "p.sr-only" })).toBeNull();
  });
});
