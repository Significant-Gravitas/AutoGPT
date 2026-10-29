import * as Dialog from "@radix-ui/react-dialog";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { WorkflowsMovedWalkthrough } from "../WorkflowsMovedWalkthrough";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (
    this: HTMLMediaElement,
  ) {
    this.dispatchEvent(new Event("play"));
    return Promise.resolve();
  });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

function getVideo() {
  return screen.getByLabelText<HTMLVideoElement>("How to find your workflows");
}

describe("WorkflowsMovedWalkthrough", () => {
  it("waits for intentional playback and provides a text alternative", () => {
    render(<WorkflowsMovedWalkthrough />);
    const video = getVideo();
    expect(video.getAttribute("src")).toBe("/videos/workflows-moved.mp4");
    expect(video.getAttribute("poster")).toBe(
      "/videos/workflows-moved-poster.webp",
    );
    expect(video.preload).toBe("metadata");
    expect(video.autoplay).toBe(false);
    expect(video.muted).toBe(true);
    expect(video.hasAttribute("playsinline")).toBe(true);
    expect(video.controls).toBe(false);
    expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();
    const description = document.getElementById(
      video.getAttribute("aria-describedby")!,
    );
    expect(description?.textContent).toContain("Team");
    expect(description?.textContent).toContain("Otto");
    expect(description?.textContent).toContain("Workflows");
    expect(description?.textContent).toContain("Select a workflow name");
    expect(description?.textContent).toContain("Setup your task");
    expect(
      screen.getByRole("button", { name: "Play walkthrough" }),
    ).toBeTruthy();
  });

  it("plays and pauses while keeping keyboard focus on the control", async () => {
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    const control = screen.getByRole("button", { name: "Play walkthrough" });
    await user.click(control);
    await screen.findByRole("button", { name: "Pause walkthrough" });
    expect(getVideo().controls).toBe(true);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(control);
    await user.keyboard(" ");
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Play walkthrough" })).toBe(
      control,
    );
    expect(document.activeElement).toBe(control);
  });

  it("shows duration from loaded metadata without guessing before it arrives", () => {
    render(<WorkflowsMovedWalkthrough />);
    expect(screen.queryByLabelText("Video duration")).toBeNull();
    Object.defineProperty(getVideo(), "duration", {
      configurable: true,
      value: Infinity,
    });
    fireEvent.loadedMetadata(getVideo());
    expect(screen.queryByLabelText("Video duration")).toBeNull();
    Object.defineProperty(getVideo(), "duration", {
      configurable: true,
      value: 26.4,
    });
    fireEvent.loadedMetadata(getVideo());
    expect(screen.getByLabelText("Video duration").textContent).toBe("0:27");
    expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();
  });

  it("offers replay after ending and restarts from the beginning", async () => {
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    getVideo().currentTime = 12;
    fireEvent.ended(getVideo());
    await user.click(
      screen.getByRole("button", { name: "Replay walkthrough" }),
    );
    expect(getVideo().currentTime).toBe(0);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
    await screen.findByRole("button", { name: "Pause walkthrough" });
  });

  it("keeps the visible control in sync with native playback controls", async () => {
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    fireEvent.pause(getVideo());
    expect(
      screen.getByRole("button", { name: "Play walkthrough" }),
    ).toBeTruthy();
    fireEvent.play(getVideo());
    expect(
      screen.getByRole("button", { name: "Pause walkthrough" }),
    ).toBeTruthy();
  });

  it("lets the user retry a rejected play request", async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(
      new Error("Playback blocked"),
    );
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    const control = screen.getByRole("button", { name: "Play walkthrough" });
    await user.click(control);
    expect((await screen.findByRole("status")).textContent).toContain(
      "Try playing it again",
    );
    expect(document.activeElement).toBe(control);
    await user.click(control);
    await screen.findByRole("button", { name: "Pause walkthrough" });
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("does not resume the custom control after a pending play is paused", async () => {
    let resolvePlay = () => {};
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(function (
      this: HTMLMediaElement,
    ) {
      this.dispatchEvent(new Event("play"));
      return new Promise<void>((resolve) => {
        resolvePlay = resolve;
      });
    });
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    fireEvent.pause(getVideo());
    await act(async () => resolvePlay());
    expect(
      screen.getByRole("button", { name: "Play walkthrough" }),
    ).toBeTruthy();
  });

  it("preserves the media fallback when a pending play later rejects", async () => {
    let rejectPlay: (reason: Error) => void = () => {};
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(
      () =>
        new Promise<void>((_, reject) => {
          rejectPlay = reject;
        }),
    );
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    fireEvent.error(getVideo());
    await act(async () => rejectPlay(new Error("Missing video")));
    expect(screen.getByRole("status").textContent).toContain("couldn't load");
    expect(screen.queryByRole("button", { name: /walkthrough/ })).toBeNull();
  });

  it("allows pausing a pending play without reporting an aborted request as an error", async () => {
    let rejectPlay: (reason: DOMException) => void = () => {};
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(function (
      this: HTMLMediaElement,
    ) {
      this.dispatchEvent(new Event("play"));
      return new Promise<void>((_, reject) => {
        rejectPlay = reject;
      });
    });
    const user = userEvent.setup();
    render(<WorkflowsMovedWalkthrough />);
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    await user.click(screen.getByRole("button", { name: "Pause walkthrough" }));
    await act(async () => rejectPlay(new DOMException("Paused", "AbortError")));
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: "Play walkthrough" }),
    ).toBeTruthy();
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("replaces failed media with directions and keeps focus in the dialog", async () => {
    const onOpenChange = vi.fn();
    const user = userEvent.setup();
    render(
      <Dialog.Root defaultOpen onOpenChange={onOpenChange}>
        <Dialog.Content aria-describedby={undefined}>
          <Dialog.Title>Your agents have moved</Dialog.Title>
          <WorkflowsMovedWalkthrough />
          <button>Show my workflows</button>
        </Dialog.Content>
      </Dialog.Root>,
    );
    await user.click(screen.getByRole("button", { name: "Play walkthrough" }));
    fireEvent.error(getVideo());
    const fallback = await screen.findByRole("status");
    expect(fallback.textContent).toContain("Team");
    expect(fallback.textContent).toContain("Otto");
    expect(fallback.textContent).toContain("Workflows");
    expect(screen.queryByRole("button", { name: /walkthrough/ })).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(fallback));
    expect(
      screen.getByRole("button", { name: "Show my workflows" }),
    ).toBeTruthy();
    expect(onOpenChange).not.toHaveBeenCalled();
  });
});
