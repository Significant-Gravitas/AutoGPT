import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TabIntroCard } from "../TabIntroCard";

const onDismiss = vi.fn();
const onCta = vi.fn();
const onAlt = vi.fn();

function card(props: { isOpen?: boolean; withAltAction?: boolean } = {}) {
  return (
    <TabIntroCard
      isOpen={props.isOpen ?? true}
      icon={SparklesIcon}
      title="Your mission control."
      body="Everything in one place."
      cta={{ label: "See my agents", onClick: onCta }}
      altAction={
        props.withAltAction
          ? { label: "Learn to build it yourself", onClick: onAlt }
          : undefined
      }
      onDismiss={onDismiss}
    />
  );
}

function renderCard(props: { isOpen?: boolean; withAltAction?: boolean } = {}) {
  return render(card(props));
}

beforeEach(() => {
  onDismiss.mockClear();
  onCta.mockClear();
  onAlt.mockClear();
});

describe("TabIntroCard", () => {
  it("renders the copy with one way forward and one way out", async () => {
    renderCard();

    expect(await screen.findByText("Your mission control.")).toBeDefined();
    expect(screen.getByText("Everything in one place.")).toBeDefined();
    expect(screen.getByRole("button", { name: "See my agents" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Got it" })).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Learn to build it yourself" }),
    ).toBeNull();
  });

  it("renders nothing while closed", () => {
    renderCard({ isOpen: false });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByText("Your mission control.")).toBeNull();
  });

  it("dismisses on Got it", async () => {
    renderCard();
    await screen.findByText("Your mission control.");

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onCta).not.toHaveBeenCalled();
  });

  it("dismisses on Escape", async () => {
    renderCard();
    await screen.findByText("Your mission control.");

    await userEvent.keyboard("{Escape}");

    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("dismisses on a click outside the card but not inside it", async () => {
    renderCard();
    await screen.findByText("Your mission control.");

    await userEvent.click(screen.getByText("Your mission control."));
    expect(onDismiss).not.toHaveBeenCalled();

    await userEvent.click(screen.getByTestId("tab-intro-overlay"));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("runs the CTA without also reporting a plain dismissal", async () => {
    renderCard();
    await screen.findByText("Your mission control.");

    await userEvent.click(
      screen.getByRole("button", { name: "See my agents" }),
    );

    expect(onCta).toHaveBeenCalledTimes(1);
    expect(onDismiss).not.toHaveBeenCalled();
  });

  it("offers the quiet alternative when the tab has one", async () => {
    renderCard({ withAltAction: true });
    await screen.findByText("Your mission control.");

    await userEvent.click(
      screen.getByRole("button", { name: "Learn to build it yourself" }),
    );

    expect(onAlt).toHaveBeenCalledTimes(1);
    expect(onDismiss).not.toHaveBeenCalled();
  });
});

describe("TabIntroCard — keyboard focus", () => {
  it("moves focus into the card on open", async () => {
    renderCard();
    await screen.findByText("Your mission control.");

    expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(
      true,
    );
  });

  it("keeps Tab and Shift+Tab inside the card", async () => {
    renderCard({ withAltAction: true });
    await screen.findByText("Your mission control.");

    const alt = screen.getByRole("button", {
      name: "Learn to build it yourself",
    });
    const gotIt = screen.getByRole("button", { name: "Got it" });
    const cta = screen.getByRole("button", { name: "See my agents" });

    await userEvent.tab();
    expect(document.activeElement).toBe(alt);
    await userEvent.tab();
    expect(document.activeElement).toBe(gotIt);
    await userEvent.tab();
    expect(document.activeElement).toBe(cta);

    // Past the last control the focus wraps back into the card rather than
    // reaching the page behind an `aria-modal` overlay.
    await userEvent.tab();
    expect(document.activeElement).toBe(alt);

    await userEvent.tab({ shift: true });
    expect(document.activeElement).toBe(cta);
  });

  it("restores focus to whatever opened it", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const { rerender } = renderCard();
    await screen.findByText("Your mission control.");
    expect(document.activeElement).not.toBe(opener);

    rerender(card({ isOpen: false }));

    await waitFor(() => expect(document.activeElement).toBe(opener));
    opener.remove();
  });
});
