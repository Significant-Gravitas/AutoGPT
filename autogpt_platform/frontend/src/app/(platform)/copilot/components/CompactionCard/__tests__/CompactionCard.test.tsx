import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CompactionCard } from "../CompactionCard";

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

  it("shows the payoff numbers when done", () => {
    render(
      <CompactionCard
        phase="done"
        stats={{
          tokensBefore: 128_000,
          tokensAfter: 31_000,
          messagesBefore: 412,
        }}
        isSettled={false}
      />,
    );
    expect(
      screen.getByText("Condensed 412 messages · 128K → 31K tokens"),
    ).toBeDefined();
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
    expect(bar.getAttribute("aria-label")).toBe("Condensing our conversation…");
  });
});
