import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DownloadSimpleIcon } from "@/components/icons/pika/adapter";

describe("pika icon adapter", () => {
  it("renders the Pika icon decoratively (no role/aria-label) by default", () => {
    const { container } = render(<DownloadSimpleIcon size={14} />);

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("viewBox")).toBe("0 0 24 24");
    expect(svg?.getAttribute("role")).toBeNull();
    expect(svg?.getAttribute("aria-label")).toBeNull();
  });

  it("labels the Pika icon when an alt is provided", () => {
    const { container } = render(
      <DownloadSimpleIcon size={14} alt="Download file" />,
    );

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("role")).toBe("img");
    expect(svg?.getAttribute("aria-label")).toBe("Download file");
  });

  it("normalizes a numeric string size", () => {
    const { container } = render(<DownloadSimpleIcon size="20" />);

    expect(container.querySelector("svg")?.getAttribute("width")).toBe("20");
  });

  it("falls back to the Pika default size when size is missing or non-numeric", () => {
    const { container: noSize } = render(<DownloadSimpleIcon />);
    expect(noSize.querySelector("svg")?.getAttribute("width")).toBe("24");

    const { container: badSize } = render(<DownloadSimpleIcon size="abc" />);
    expect(badSize.querySelector("svg")?.getAttribute("width")).toBe("24");
  });
});
