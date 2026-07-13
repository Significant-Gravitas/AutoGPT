import { render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const useGetFlagMock = vi.fn<(flag: string) => boolean>();

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { PIKA_ICONS: "pika-icons" },
  useGetFlag: (flag: string) => useGetFlagMock(flag),
}));

import { DownloadSimpleIcon } from "@/components/icons/pika/adapter";

afterEach(() => {
  useGetFlagMock.mockReset();
});

describe("pika icon adapter", () => {
  it("renders the Phosphor icon when the flag is off", () => {
    useGetFlagMock.mockReturnValue(false);
    const { container } = render(<DownloadSimpleIcon size={14} />);

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("viewBox")).toBe("0 0 256 256");
  });

  it("renders the Pika icon decoratively (no role/aria-label) when the flag is on", () => {
    useGetFlagMock.mockReturnValue(true);
    const { container } = render(<DownloadSimpleIcon size={14} />);

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("viewBox")).toBe("0 0 24 24");
    expect(svg?.getAttribute("role")).toBeNull();
    expect(svg?.getAttribute("aria-label")).toBeNull();
  });

  it("labels the Pika icon when an alt is provided", () => {
    useGetFlagMock.mockReturnValue(true);
    const { container } = render(
      <DownloadSimpleIcon size={14} alt="Download file" />,
    );

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("role")).toBe("img");
    expect(svg?.getAttribute("aria-label")).toBe("Download file");
  });

  it("normalizes a numeric string size", () => {
    useGetFlagMock.mockReturnValue(true);
    const { container } = render(<DownloadSimpleIcon size="20" />);

    expect(container.querySelector("svg")?.getAttribute("width")).toBe("20");
  });

  it("falls back to the Pika default size when size is missing or non-numeric", () => {
    useGetFlagMock.mockReturnValue(true);
    const { container: noSize } = render(<DownloadSimpleIcon />);
    expect(noSize.querySelector("svg")?.getAttribute("width")).toBe("24");

    const { container: badSize } = render(<DownloadSimpleIcon size="abc" />);
    expect(badSize.querySelector("svg")?.getAttribute("width")).toBe("24");
  });
});
