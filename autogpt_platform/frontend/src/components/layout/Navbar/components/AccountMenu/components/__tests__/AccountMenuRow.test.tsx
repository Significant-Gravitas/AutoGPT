import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { AccountMenuRow } from "../AccountMenuRow";

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

describe("AccountMenuRow", () => {
  test("renders as internal link with href when as=link", () => {
    render(
      <AccountMenuRow
        as="link"
        href="/settings"
        icon={<span data-testid="icon" />}
        label="Settings"
      />,
    );

    const link = screen.getByText("Settings").closest("a");
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href")).toBe("/settings");
    expect(link?.getAttribute("target")).toBeNull();
    expect(screen.getByTestId("icon")).toBeDefined();
  });

  test("renders as external anchor with target=_blank when external=true", () => {
    render(
      <AccountMenuRow
        as="link"
        href="https://example.com"
        external
        icon={<span />}
        label="Docs"
      />,
    );

    const anchor = screen.getByText("Docs").closest("a");
    expect(anchor?.getAttribute("target")).toBe("_blank");
    expect(anchor?.getAttribute("rel")).toBe("noopener noreferrer");
  });

  test("renders as button when as=button and triggers onClick", () => {
    const onClick = vi.fn();
    render(
      <AccountMenuRow
        as="button"
        onClick={onClick}
        icon={<span />}
        label="Publish"
      />,
    );

    const button = screen.getByRole("button", { name: /publish/i });
    expect(button.tagName).toBe("BUTTON");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  test("falls back to button when as=link but no href provided", () => {
    const onClick = vi.fn();
    render(
      <AccountMenuRow
        as="link"
        onClick={onClick}
        icon={<span />}
        label="No href"
      />,
    );

    const button = screen.getByRole("button", { name: /no href/i });
    expect(button.tagName).toBe("BUTTON");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalled();
  });

  test("applies destructive classes when destructive=true", () => {
    render(
      <AccountMenuRow
        as="button"
        destructive
        icon={<span />}
        label="Log out"
      />,
    );

    const button = screen.getByRole("button", { name: /log out/i });
    expect(button.className).toContain("hover:bg-red-50");
  });

  test("shows pending spinner when useLinkStatus returns pending", async () => {
    vi.resetModules();
    vi.doMock("next/link", () => ({
      __esModule: true,
      default: ({ children, href, ...props }: any) => (
        <a href={href} {...props}>
          {children}
        </a>
      ),
      useLinkStatus: () => ({ pending: true }),
    }));

    const { AccountMenuRow: PendingRow } = await import("../AccountMenuRow");

    render(
      <PendingRow as="link" href="/foo" icon={<span />} label="Loading link" />,
    );

    const link = screen.getByText("Loading link").closest("a");
    expect(link?.querySelector("svg")).not.toBeNull();
    vi.doUnmock("next/link");
  });
});
