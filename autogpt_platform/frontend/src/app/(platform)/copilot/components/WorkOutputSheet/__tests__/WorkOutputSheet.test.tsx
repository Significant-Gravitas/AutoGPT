import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { WorkOutputSheet } from "../WorkOutputSheet";

const { detailsResult } = vi.hoisted(() => ({
  detailsResult: {
    current: {
      data: null as unknown,
      isLoading: false,
      isError: false,
    },
  },
}));

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetExecutionDetails: () => detailsResult.current,
}));

function setOutputs(outputs: Record<string, unknown[]>) {
  detailsResult.current = {
    data: { outputs },
    isLoading: false,
    isError: false,
  };
}

const baseProps = {
  open: true,
  onOpenChange: () => {},
  title: "Weekly Report",
  graphId: "graph-1",
  executionId: "exec-1",
  runLink: "/library/agents/lib-1?activeTab=runs&activeItem=exec-1",
};

beforeEach(() => {
  detailsResult.current = { data: null, isLoading: false, isError: false };
});

describe("WorkOutputSheet", () => {
  it("renders a table with a CSV export for table output", () => {
    setOutputs({ result: [[{ metric: "signups", value: 12 }]] });
    render(<WorkOutputSheet {...baseProps} outputType="table" />);

    expect(screen.getByRole("button", { name: "Export CSV" })).toBeDefined();
    expect(screen.getByText("metric")).toBeDefined();
    expect(screen.getByText("signups")).toBeDefined();
  });

  it("opens the classified pin even when an unrenderable pin comes first", () => {
    setOutputs({
      status: ["ok"],
      results: [[{ metric: "signups", value: 12 }]],
    });
    render(
      <WorkOutputSheet {...baseProps} outputType="table" outputKey="results" />,
    );

    expect(screen.getByText("metric")).toBeDefined();
    expect(screen.getByText("signups")).toBeDefined();
  });

  it("caps large tables and shows a truncation notice", () => {
    const rows = Array.from({ length: 150 }, (_, i) => ({ index: i }));
    setOutputs({ result: [rows] });
    render(<WorkOutputSheet {...baseProps} outputType="table" />);

    expect(screen.getByText("99")).toBeDefined();
    expect(screen.queryByText("120")).toBeNull();
    expect(screen.getByText(/Showing the first 100 of 150 rows/)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "open the full run" }),
    ).toBeDefined();
  });

  it("renders markdown for doc output", () => {
    setOutputs({ result: ["# Heading text"] });
    render(<WorkOutputSheet {...baseProps} outputType="doc" />);

    expect(screen.getByText("Heading text")).toBeDefined();
  });

  it("renders an image for image output", () => {
    setOutputs({ result: ["https://cdn.example.com/chart.png"] });
    render(<WorkOutputSheet {...baseProps} outputType="image" />);

    const image = screen.getByAltText("Weekly Report") as HTMLImageElement;
    expect(image.src).toBe("https://cdn.example.com/chart.png");
  });

  it("falls back to the run link for unknown output", () => {
    render(<WorkOutputSheet {...baseProps} outputType="unknown" />);

    const link = screen.getByRole("link", { name: "Open run details" });
    expect(link.getAttribute("href")).toBe(baseProps.runLink);
  });
});
