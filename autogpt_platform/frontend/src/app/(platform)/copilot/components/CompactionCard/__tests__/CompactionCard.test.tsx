import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CompactionCard } from "../CompactionCard";

function preferReducedMotion() {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: query.includes("prefers-reduced-motion"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CompactionCard", () => {
  it("narrates the summarizing phase with a live progress bar", () => {
    render(<CompactionCard phase="summarizing" stats={{}} isSettled={false} />);
    expect(screen.getByText("Condensing our conversation…")).toBeDefined();
    expect(screen.getByRole("progressbar")).toBeDefined();
  });

  it("narrates the rebuilding phase", () => {
    render(
      <CompactionCard
        phase="rebuilding"
        stats={{ tokensBefore: 128_000 }}
        isSettled={false}
      />,
    );
    expect(screen.getByText("Reloading context…")).toBeDefined();
  });

  it("shows the payoff numbers once settled", () => {
    render(
      <CompactionCard
        phase={null}
        stats={{
          tokensBefore: 128_000,
          tokensAfter: 31_000,
          messagesBefore: 412,
          messagesAfter: 38,
        }}
        isSettled
      />,
    );
    expect(
      screen.getByText("Condensed 412 messages · 128K → 31K tokens"),
    ).toBeDefined();
  });

  it("treats a live row with no phase yet as the opening state", () => {
    // Between `tool-input-start` and the first `data-compaction` chunk the
    // row is live but phaseless — it must read and animate as summarizing,
    // never as finished copy over a frozen bar.
    render(<CompactionCard phase={null} stats={{}} isSettled={false} />);
    expect(screen.getByText("Condensing our conversation…")).toBeDefined();
    expect(screen.getByRole("progressbar")).toBeDefined();
    expect(
      screen.queryByText("Condensed the conversation to keep going"),
    ).toBeNull();
  });

  it("drops the bar once settled", () => {
    render(
      <CompactionCard
        phase={null}
        stats={{ tokensBefore: 128_000, tokensAfter: 31_000 }}
        isSettled
      />,
    );
    expect(
      screen.getByText("Condensed the conversation · 128K → 31K tokens"),
    ).toBeDefined();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("reports a reset, not a condensation, when history was dropped", () => {
    render(
      <CompactionCard
        phase={null}
        stats={{ dropped: true, messagesBefore: 412 }}
        isSettled
      />,
    );
    expect(
      screen.getByText(
        "Started a fresh context — earlier messages were dropped",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/Condensed/)).toBeNull();
  });

  it("degrades to plain copy for legacy rows without stats", () => {
    render(<CompactionCard phase={null} stats={{}} isSettled />);
    expect(
      screen.getByText("Condensed the conversation to keep going"),
    ).toBeDefined();
  });

  it("exposes the progress value to assistive tech", () => {
    render(<CompactionCard phase="summarizing" stats={{}} isSettled={false} />);
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuemin")).toBe("0");
    expect(bar.getAttribute("aria-valuemax")).toBe("100");
    expect(bar.getAttribute("aria-valuenow")).toBe("2");
    expect(bar.getAttribute("aria-label")).toBe("Condensing our conversation…");
  });

  it("reports the token drop when compression summarized rows in place", () => {
    // What QA actually sees on a live run: the same message count in and
    // out, because compression rewrites rows rather than removing them. The
    // token half is the whole payoff, and it has to render.
    render(
      <CompactionCard
        phase={null}
        stats={{
          tokensBefore: 233_000,
          tokensAfter: 97_000,
          messagesBefore: 36,
          messagesAfter: 36,
        }}
        isSettled
      />,
    );
    expect(
      screen.getByText("Condensed the conversation · 233K → 97K tokens"),
    ).toBeDefined();
  });

  it("steps the bar under reduced motion but keeps aria-valuenow exact", () => {
    preferReducedMotion();
    render(<CompactionCard phase="summarizing" stats={{}} isSettled={false} />);
    const bar = screen.getByRole("progressbar");
    // The opening 2% is below the first step, so the fill is still empty
    // while the value assistive tech reads stays truthful.
    expect(bar.getAttribute("aria-valuenow")).toBe("2");
    expect((bar.firstElementChild as HTMLElement).style.width).toBe("0%");
  });

  it("paints the exact percent when motion is allowed", () => {
    render(<CompactionCard phase="summarizing" stats={{}} isSettled={false} />);
    const bar = screen.getByRole("progressbar");
    expect((bar.firstElementChild as HTMLElement).style.width).toBe("2%");
  });

  it("announces label changes politely without turning the bar into a live region", () => {
    render(<CompactionCard phase="summarizing" stats={{}} isSettled={false} />);
    const label = screen.getByText("Condensing our conversation…");
    expect(label.getAttribute("aria-live")).toBe("polite");
    expect(
      screen.getByRole("progressbar").getAttribute("aria-live"),
    ).toBeNull();
  });
});
