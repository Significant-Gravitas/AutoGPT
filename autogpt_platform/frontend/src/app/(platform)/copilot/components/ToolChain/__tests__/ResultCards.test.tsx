import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  ChipList,
  LinkCard,
  StatCard,
  StatusCard,
  StatusPill,
} from "../ResultCards";

describe("StatusPill", () => {
  afterEach(cleanup);

  it("normalizes known statuses to lowercase text", () => {
    render(<StatusPill status="Completed" />);

    expect(screen.getByText("completed")).toBeDefined();
  });

  it("renders unknown statuses with the default style", () => {
    render(<StatusPill status="paused" />);

    expect(screen.getByText("paused")).toBeDefined();
  });
});

describe("StatusCard", () => {
  afterEach(cleanup);

  it("renders the label for successful states", () => {
    render(<StatusCard ok label="Everything worked" />);

    expect(screen.getByText("Everything worked")).toBeDefined();
  });

  it("renders the label for failed states", () => {
    render(<StatusCard ok={false} label="Something broke" />);

    expect(screen.getByText("Something broke")).toBeDefined();
  });
});

describe("StatCard", () => {
  afterEach(cleanup);

  it("renders the value and label", () => {
    render(<StatCard value={5} label="deleted items" />);

    expect(screen.getByText("5")).toBeDefined();
    expect(screen.getByText("deleted items")).toBeDefined();
  });
});

describe("ChipList", () => {
  afterEach(cleanup);

  it("renders a label with deduplicated chips", () => {
    render(<ChipList label="Tags" items={["alpha", "alpha", "beta"]} />);

    expect(screen.getByText("Tags")).toBeDefined();
    expect(screen.getAllByText("alpha")).toHaveLength(1);
    expect(screen.getByText("beta")).toBeDefined();
  });
});

describe("LinkCard", () => {
  afterEach(cleanup);

  it("shows the title above the domain with meta text", () => {
    render(
      <LinkCard
        url="https://www.example.com/page"
        title="Example page"
        meta="12.0 KB"
      />,
    );

    expect(screen.getByText("Example page")).toBeDefined();
    expect(screen.getByText("example.com")).toBeDefined();
    expect(screen.getByText("12.0 KB")).toBeDefined();
    expect(screen.getByLabelText("Open link").getAttribute("href")).toBe(
      "https://www.example.com/page",
    );
  });

  it("falls back to the domain as title and shows the raw URL", () => {
    render(<LinkCard url="https://example.com/deep/path" />);

    expect(screen.getByText("example.com")).toBeDefined();
    expect(screen.getByText("https://example.com/deep/path")).toBeDefined();
  });

  it("uses the raw URL as the display domain when it cannot be parsed", () => {
    render(<LinkCard url="not-a-url" />);

    expect(screen.getAllByText("not-a-url").length).toBeGreaterThan(0);
  });
});
