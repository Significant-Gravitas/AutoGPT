import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  DocsList,
  FeatureRequestList,
  FileList,
  FolderList,
  ScheduleCreatedCard,
  ScheduleList,
} from "../ListCards";

describe("FeatureRequestList", () => {
  afterEach(cleanup);

  it("renders titles, descriptions and identifiers", () => {
    render(
      <FeatureRequestList
        results={[
          {
            title: "Dark mode",
            description: "Please add it",
            identifier: "FR-101",
          },
          { note: "untitled" },
        ]}
      />,
    );

    expect(screen.getByText("Dark mode")).toBeDefined();
    expect(screen.getByText("Please add it")).toBeDefined();
    expect(screen.getByText("FR-101")).toBeDefined();
    expect(screen.getByText('{"note":"untitled"}')).toBeDefined();
  });
});

describe("ScheduleList", () => {
  afterEach(cleanup);

  it("labels copilot turn schedules as chat", () => {
    render(
      <ScheduleList
        schedules={[
          {
            name: "Daily briefing",
            next_run_time: "2026-08-21T10:00:00Z",
            cron: "0 10 * * *",
            kind: "copilot_turn",
          },
        ]}
      />,
    );

    expect(screen.getByText("Daily briefing")).toBeDefined();
    expect(screen.getByText("chat")).toBeDefined();
  });

  it("labels other schedule kinds as agent", () => {
    render(
      <ScheduleList
        schedules={[{ message: "Run scraper", kind: "graph_execution" }]}
      />,
    );

    expect(screen.getByText("Run scraper")).toBeDefined();
    expect(screen.getByText("agent")).toBeDefined();
  });

  it("falls back to inline JSON for unnamed schedules", () => {
    render(<ScheduleList schedules={[{ enabled: true }]} />);

    expect(screen.getByText('{"enabled":true}')).toBeDefined();
  });
});

describe("ScheduleCreatedCard", () => {
  afterEach(cleanup);

  it("announces the scheduled follow-up", () => {
    render(
      <ScheduleCreatedCard
        output={{ next_run_time: "2026-08-21T10:00:00Z", is_recurring: true }}
      />,
    );

    expect(screen.getByText("Follow-up scheduled")).toBeDefined();
  });
});

describe("FolderList", () => {
  afterEach(cleanup);

  it("renders folder names with singular and plural agent counts", () => {
    render(
      <FolderList
        folders={[
          { name: "Marketing", agent_count: 1 },
          { name: "Research", agent_count: 3 },
          { name: "Empty" },
        ]}
      />,
    );

    expect(screen.getByText("Marketing")).toBeDefined();
    expect(screen.getByText("1 agent")).toBeDefined();
    expect(screen.getByText("Research")).toBeDefined();
    expect(screen.getByText("3 agents")).toBeDefined();
    expect(screen.getByText("Empty")).toBeDefined();
  });
});

describe("FileList", () => {
  afterEach(cleanup);

  it("renders file paths with sizes", () => {
    render(
      <FileList
        files={[
          { path: "chart.png", mime_type: "image/png", size_bytes: 2048 },
          { name: "notes.txt", size_bytes: 12 },
        ]}
      />,
    );

    expect(screen.getByText("chart.png")).toBeDefined();
    expect(screen.getByText("2.0 KB")).toBeDefined();
    expect(screen.getByText("notes.txt")).toBeDefined();
    expect(screen.getByText("12 B")).toBeDefined();
  });
});

describe("DocsList", () => {
  afterEach(cleanup);

  it("renders titles, sections, snippets and links", () => {
    render(
      <DocsList
        results={[
          {
            title: "Blocks",
            section: "Guide",
            snippet: "How blocks work",
            doc_url: "https://docs.agpt.co/blocks",
          },
        ]}
      />,
    );

    expect(screen.getByText("Blocks")).toBeDefined();
    expect(screen.getByText("Guide")).toBeDefined();
    expect(screen.getByText("How blocks work")).toBeDefined();
    expect(screen.getByLabelText("Open doc").getAttribute("href")).toBe(
      "https://docs.agpt.co/blocks",
    );
  });

  it("falls back to the path and omits the link without a URL", () => {
    render(<DocsList results={[{ path: "getting-started.md" }]} />);

    expect(screen.getByText("getting-started.md")).toBeDefined();
    expect(screen.queryByLabelText("Open doc")).toBeNull();
  });
});
