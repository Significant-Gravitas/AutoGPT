import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { connected } from "@/lib/openui/__tests__/connected-fixtures";
import { showResult } from "./renderConnected";

function pointer(
  target: HTMLElement,
  type: string,
  x: number,
  y = 100,
  id = 1,
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    clientX: x,
    clientY: y,
    button: 0,
  });
  Object.defineProperties(event, {
    pointerId: { value: id },
    pointerType: { value: "touch" },
    isPrimary: { value: id === 1 },
  });
  fireEvent(target, event);
}

describe("A/B comparison gestures and accessible alternatives", () => {
  beforeEach(() => sessionStorage.clear());

  it("swipes left for A and right for B, does not send, and supports undo", async () => {
    const send = vi.fn();
    await showResult(send);
    const board = await screen.findByRole("group", {
      name: /swipe comparison/,
    });
    pointer(board, "pointerdown", 200);
    pointer(board, "pointermove", 90);
    pointer(board, "pointerup", 90);
    const a = screen.getByRole("button", { name: "Choose City hotel" });
    expect(a.getAttribute("aria-pressed")).toBe("true");
    pointer(board, "pointerdown", 90);
    pointer(board, "pointermove", 200);
    pointer(board, "pointerup", 200);
    expect(
      screen
        .getByRole("button", { name: "Choose Coastal cabin" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    expect(send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Undo choice" }));
    expect(a.getAttribute("aria-pressed")).toBe("true");
  });

  it.each(["short", "vertical", "cancelled", "second finger"])(
    "ignores a %s gesture",
    async (kind) => {
      await showResult();
      const board = await screen.findByRole("group", {
        name: /swipe comparison/,
      });
      pointer(board, "pointerdown", 200, 100, kind === "second finger" ? 2 : 1);
      pointer(
        board,
        "pointermove",
        kind === "short" ? 175 : 80,
        kind === "vertical" ? 350 : 100,
      );
      pointer(
        board,
        kind === "cancelled" ? "pointercancel" : "pointerup",
        kind === "short" ? 175 : 80,
        kind === "vertical" ? 350 : 100,
      );
      expect(
        screen
          .getByRole("button", { name: "Choose City hotel" })
          .getAttribute("aria-pressed"),
      ).toBe("false");
      expect(screen.queryByRole("button", { name: "Undo choice" })).toBeNull();
    },
  );

  it("does not swipe a shared view, and normal keyboard buttons work in a private view", async () => {
    const shared = await showResult(vi.fn(), true);
    const board = await screen.findByRole("group", {
      name: /swipe comparison/,
    });
    pointer(board, "pointerdown", 200);
    pointer(board, "pointerup", 80);
    expect(
      screen
        .getByRole("button", { name: "Choose City hotel" })
        .getAttribute("aria-pressed"),
    ).toBe("false");
    shared.unmount();
    await showResult();
    const option = await screen.findByRole("button", {
      name: "Choose City hotel",
    });
    option.focus();
    await userEvent.keyboard("{Enter}");
    expect(option.getAttribute("aria-pressed")).toBe("true");
  });

  it("permits blank optional notes, but enforces required multi-selects", async () => {
    const send = vi.fn();
    await showResult(
      send,
      false,
      connected
        .replace('["art"], [{label:', "[], [{label:")
        .replace('value:"history"}], false)', 'value:"history"}], true)'),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose City hotel" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    expect(send).not.toHaveBeenCalled();
    expect(
      await screen.findByText("Choose at least one option."),
    ).toBeDefined();
    fireEvent.click(screen.getByLabelText("Art"));
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    expect(send).toHaveBeenCalledTimes(1);
    expect(send.mock.calls[0][0]).toContain('"notes": ""');
  });
});
