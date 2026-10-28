import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { ComposerOverflowMenu } from "../ComposerOverflowMenu";

function openMenu() {
  fireEvent.click(screen.getByTestId("composer-overflow-button"));
}

describe("ComposerOverflowMenu", () => {
  it("keeps its controls tucked away until opened", () => {
    render(
      <ComposerOverflowMenu>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );

    expect(screen.queryByText("Inner control")).toBeNull();
    expect(screen.getByRole("button", { name: "More options" })).toBeDefined();
  });

  it("reveals its controls when the button is pressed", async () => {
    render(
      <ComposerOverflowMenu>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    openMenu();

    expect(await screen.findByText("Inner control")).toBeDefined();
    expect(screen.getByTestId("composer-overflow-panel")).toBeDefined();
  });

  it("stays shut while forced closed", () => {
    render(
      <ComposerOverflowMenu forceClosed>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    openMenu();

    expect(screen.queryByText("Inner control")).toBeNull();
  });

  it("cannot be opened while disabled", () => {
    render(
      <ComposerOverflowMenu disabled>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    openMenu();

    expect(screen.queryByText("Inner control")).toBeNull();
  });

  it("does not reopen on its own after being forced closed", async () => {
    const { rerender } = render(
      <ComposerOverflowMenu>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    openMenu();
    expect(await screen.findByText("Inner control")).toBeDefined();

    rerender(
      <ComposerOverflowMenu forceClosed>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    expect(screen.queryByText("Inner control")).toBeNull();

    rerender(
      <ComposerOverflowMenu>
        <button>Inner control</button>
      </ComposerOverflowMenu>,
    );
    expect(screen.queryByText("Inner control")).toBeNull();
  });
});
