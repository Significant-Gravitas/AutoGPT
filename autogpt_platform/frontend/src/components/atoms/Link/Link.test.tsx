import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { forwardRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { Link } from "./Link";

// Mock Next.js Link with proper ref forwarding
vi.mock("next/link", () => ({
  default: forwardRef(function MockNextLink(
    { href, children, ...props }: any,
    ref: any,
  ) {
    return (
      <a ref={ref} href={href} {...props}>
        {children}
      </a>
    );
  }),
}));

describe("Link Component", () => {
  it("renders internal link with correct href", () => {
    render(<Link href="/dashboard">Dashboard</Link>);

    const link = screen.getByRole("link", { name: "Dashboard" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/dashboard");
  });

  it("renders external link with target blank", () => {
    render(
      <Link href="https://example.com" isExternal>
        External Link
      </Link>,
    );

    const link = screen.getByRole("link", { name: "External Link" });
    expect(link).toHaveAttribute("href", "https://example.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("handles click events", async () => {
    const handleClick = vi.fn((e) => e.preventDefault());
    const user = userEvent.setup();

    render(
      <Link href="/dashboard" onClick={handleClick}>
        Dashboard
      </Link>,
    );

    const link = screen.getByRole("link", { name: "Dashboard" });
    await user.click(link);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it("supports keyboard navigation", async () => {
    const handleKeyDown = vi.fn((e) => {
      if (e.key === "Enter") e.preventDefault();
    });
    const user = userEvent.setup();

    render(
      <Link href="/dashboard" onKeyDown={handleKeyDown}>
        Dashboard
      </Link>,
    );

    const link = screen.getByRole("link", { name: "Dashboard" });

    await user.tab();
    expect(link).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(handleKeyDown).toHaveBeenCalled();
  });

  it("applies custom className", () => {
    render(
      <Link href="/dashboard" className="custom-class">
        Dashboard
      </Link>,
    );

    const link = screen.getByRole("link", { name: "Dashboard" });
    expect(link).toHaveClass("custom-class");
  });

  it("forwards ref correctly", () => {
    const ref = { current: null };

    render(
      <Link href="/dashboard" ref={ref}>
        Dashboard
      </Link>,
    );

    // Check that ref is populated
    expect(ref.current).toBeTruthy();
  });

  it("passes through additional props", () => {
    render(
      <Link href="/dashboard" data-testid="custom-link">
        Dashboard
      </Link>,
    );

    const link = screen.getByRole("link", { name: "Dashboard" });
    expect(link).toHaveAttribute("data-testid", "custom-link");
  });

  it("renders children correctly", () => {
    render(
      <Link href="/dashboard">
        <span>Dashboard</span>
        <span>Icon</span>
      </Link>,
    );

    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Icon")).toBeInTheDocument();
  });

  it("distinguishes between internal and external links", () => {
    const { rerender } = render(<Link href="/internal">Internal</Link>);

    let link = screen.getByRole("link", { name: "Internal" });
    expect(link).not.toHaveAttribute("target");
    expect(link).not.toHaveAttribute("rel");

    rerender(
      <Link href="https://external.com" isExternal>
        External
      </Link>,
    );

    link = screen.getByRole("link", { name: "External" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });
});
