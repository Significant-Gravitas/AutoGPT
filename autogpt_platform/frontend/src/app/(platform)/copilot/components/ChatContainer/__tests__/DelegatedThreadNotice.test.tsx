import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DelegatedThreadNotice } from "../components/DelegatedThreadNotice";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

describe("DelegatedThreadNotice", () => {
  afterEach(cleanup);

  it("says Otto delegated the thread and links back to Otto's thread", () => {
    render(
      <DelegatedThreadNotice
        sentFrom={{ sessionId: "parent-1", expertId: null, expertName: null }}
        expertName="Alex"
      />,
    );
    const notice = screen.getByTestId("delegated-thread-notice");
    expect(notice.textContent).toContain("Delegated by Otto");
    expect(notice.textContent).toContain("Alex is working for Otto");
    expect(notice.textContent).toContain("you own the outcome");
    expect(
      screen
        .getByRole("link", { name: /back to otto's thread/i })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=parent-1");
  });

  it("names the expert who sent the task when it was not Otto", () => {
    render(
      <DelegatedThreadNotice
        sentFrom={{
          sessionId: "parent-2",
          expertId: "exp-devon",
          expertName: "Devon",
        }}
        expertName={null}
      />,
    );
    const notice = screen.getByTestId("delegated-thread-notice");
    expect(notice.textContent).toContain("Delegated by Devon");
    expect(notice.textContent).not.toContain("you own the outcome");
  });
});
