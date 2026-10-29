import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MicBlockedNotice } from "../MicBlockedNotice";

const BLOCKED =
  "Your browser blocked the microphone, so write here instead. To talk, allow the microphone in your browser's site settings (the lock or microphone icon by the address bar), then try the microphone again.";
const DISMISSED =
  "The microphone prompt was closed, so write here instead. To talk, try the microphone again.";
const SYSTEM =
  "Your computer blocked the microphone, so write here instead. To talk, let your browser use the microphone in your computer's privacy settings, then try the microphone again.";

describe("MicBlockedNotice", () => {
  it.each([
    ["denied", BLOCKED],
    ["unknown", BLOCKED],
    ["prompt", DISMISSED],
    ["granted", SYSTEM],
  ] as const)(
    "explains a %s microphone and how to get it back",
    (permission, copy) => {
      render(
        <MicBlockedNotice
          isBlocked
          permission={permission}
          onRetry={vi.fn()}
        />,
      );

      expect(screen.getByRole("alert").textContent).toBe(copy);
    },
  );

  it("asks for the microphone again from its link", async () => {
    const onRetry = vi.fn();
    render(
      <MicBlockedNotice isBlocked permission="denied" onRetry={onRetry} />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "try the microphone again" }),
    );

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("updates its copy in place when the permission changes", () => {
    const { rerender } = render(
      <MicBlockedNotice isBlocked permission="prompt" onRetry={vi.fn()} />,
    );
    const alert = screen.getByRole("alert");

    rerender(
      <MicBlockedNotice isBlocked permission="denied" onRetry={vi.fn()} />,
    );

    expect(screen.getAllByRole("alert")).toEqual([alert]);
    expect(alert.textContent).toBe(BLOCKED);
  });

  it("explains a microphone that could not be opened", () => {
    render(
      <MicBlockedNotice
        isBlocked={false}
        isUnavailable
        permission="granted"
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByRole("alert").textContent).toBe(
      "We couldn't open your microphone, so write here instead. To talk, check that it's connected and no other app is using it, then try the microphone again.",
    );
  });

  it("renders nothing while the microphone is not blocked", () => {
    const { container } = render(
      <MicBlockedNotice
        isBlocked={false}
        permission="denied"
        onRetry={vi.fn()}
      />,
    );

    expect(container.innerHTML).toBe("");
  });
});
