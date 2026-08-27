import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  BlockListCard,
  BlockOutputCard,
  CardProviderIcon,
} from "../BlockCards";

describe("CardProviderIcon", () => {
  afterEach(cleanup);

  it("renders the integration icon for a provider", () => {
    render(<CardProviderIcon provider="github" fallback={<span>fb</span>} />);

    const img = screen.getByAltText("github");
    expect(img.getAttribute("src")).toBe("/integrations/github.png");
  });

  it("renders the fallback without a provider", () => {
    render(<CardProviderIcon provider={null} fallback={<span>fb</span>} />);

    expect(screen.getByText("fb")).toBeDefined();
  });

  it("renders the fallback for providers without safe characters", () => {
    render(<CardProviderIcon provider="../../" fallback={<span>fb</span>} />);

    expect(screen.getByText("fb")).toBeDefined();
  });

  it("falls back after the image fails to load", () => {
    render(<CardProviderIcon provider="github" fallback={<span>fb</span>} />);

    fireEvent.error(screen.getByAltText("github"));

    expect(screen.getByText("fb")).toBeDefined();
    expect(screen.queryByAltText("github")).toBeNull();
  });
});

describe("BlockListCard", () => {
  afterEach(cleanup);

  it("renders block names, descriptions and lowercased category chips", () => {
    render(
      <BlockListCard
        blocks={[
          {
            name: "HTTP Request",
            description: "Makes requests",
            provider: "github",
            categories: ["NETWORK", "DATA"],
          },
          { block_name: "Send Email" },
        ]}
      />,
    );

    expect(screen.getByText("HTTP Request")).toBeDefined();
    expect(screen.getByText("Makes requests")).toBeDefined();
    expect(screen.getByText("network")).toBeDefined();
    expect(screen.queryByText("data")).toBeNull();
    expect(screen.getByText("Send Email")).toBeDefined();
    expect(screen.getByAltText("github")).toBeDefined();
  });

  it("falls back to inline JSON for unnamed blocks", () => {
    render(<BlockListCard blocks={[{ enabled: true }]} />);

    expect(screen.getByText('{"enabled":true}')).toBeDefined();
  });
});

describe("BlockOutputCard", () => {
  afterEach(cleanup);

  it("renders the block name with flattened single-value outputs", () => {
    render(
      <BlockOutputCard
        output={{
          block_name: "Send Email",
          outputs: { message_id: ["msg-1"], recipients: ["a", "b"] },
        }}
      />,
    );

    expect(screen.getByText("Send Email")).toBeDefined();
    expect(screen.getByText("message id")).toBeDefined();
    expect(screen.getByText("msg-1")).toBeDefined();
    expect(screen.getByText("recipients")).toBeDefined();
    expect(screen.getByText('["a","b"]')).toBeDefined();
  });

  it("falls back to the block id without outputs", () => {
    render(
      <BlockOutputCard output={{ block_id: "block-9", success: false }} />,
    );

    expect(screen.getByText("block-9")).toBeDefined();
  });
});
