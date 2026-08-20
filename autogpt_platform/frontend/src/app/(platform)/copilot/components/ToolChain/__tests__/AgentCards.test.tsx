import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  AgentListCard,
  AgentPreviewCard,
  AgentSavedCard,
  SubSessionCard,
} from "../AgentCards";

describe("AgentListCard", () => {
  afterEach(cleanup);

  it("links library agents to the library page", () => {
    render(
      <AgentListCard
        agents={[{ id: "lib-1", source: "library", name: "Lib Agent" }]}
      />,
    );

    expect(screen.getByText("Lib Agent")).toBeDefined();
    expect(screen.getByLabelText("Open agent").getAttribute("href")).toBe(
      "/library/agents/lib-1",
    );
  });

  it("links marketplace agents by creator and slug", () => {
    render(
      <AgentListCard
        agents={[
          {
            id: "creator/scraper",
            name: "Scraper",
            creator: "creator",
            description: "Scrapes websites",
            runs: 1200,
            rating: 4.55,
          },
        ]}
      />,
    );

    expect(screen.getByText("Scraper")).toBeDefined();
    expect(screen.getByText("by creator")).toBeDefined();
    expect(screen.getByText("Scrapes websites")).toBeDefined();
    expect(screen.getByText("1,200 runs")).toBeDefined();
    expect(screen.getByText("4.5")).toBeDefined();
    expect(screen.getByLabelText("Open agent").getAttribute("href")).toBe(
      "/marketplace/agent/creator/scraper",
    );
  });

  it("omits the link for ids that are not creator/slug pairs", () => {
    render(
      <AgentListCard agents={[{ id: "a/b/c", name: "Deep Path Agent" }]} />,
    );

    expect(screen.getByText("Deep Path Agent")).toBeDefined();
    expect(screen.queryByLabelText("Open agent")).toBeNull();
  });

  it("falls back to inline JSON for unnamed agents", () => {
    render(<AgentListCard agents={[{ runs: 3 }]} />);

    expect(screen.getByText('{"runs":3}')).toBeDefined();
  });
});

describe("AgentSavedCard", () => {
  afterEach(cleanup);

  it("shows the version with builder and library links", () => {
    render(
      <AgentSavedCard
        output={{
          agent_name: "My Agent",
          graph_version: 3,
          agent_page_link: "/build?flowID=graph-1",
          library_agent_link: "/library/agents/lib-1",
        }}
      />,
    );

    expect(screen.getByText("My Agent")).toBeDefined();
    expect(screen.getByText("v3")).toBeDefined();
    expect(screen.getByLabelText("Open in builder").getAttribute("href")).toBe(
      "/build?flowID=graph-1",
    );
    expect(screen.getByLabelText("Open in library").getAttribute("href")).toBe(
      "/library/agents/lib-1",
    );
  });

  it("falls back to a default name and no links", () => {
    render(<AgentSavedCard output={{}} />);

    expect(screen.getByText("Agent")).toBeDefined();
    expect(screen.queryByLabelText("Open in library")).toBeNull();
  });

  it("builds the library link from the library agent id", () => {
    render(
      <AgentSavedCard
        output={{ name: "Fallback Agent", library_agent_id: "lib-9" }}
      />,
    );

    expect(screen.getByText("Fallback Agent")).toBeDefined();
    expect(screen.getByLabelText("Open in library").getAttribute("href")).toBe(
      "/library/agents/lib-9",
    );
    expect(screen.queryByLabelText("Open in builder")).toBeNull();
  });
});

describe("AgentPreviewCard", () => {
  afterEach(cleanup);

  it("shows the name, description and block count", () => {
    render(
      <AgentPreviewCard
        output={{
          agent_name: "Draft agent",
          description: "Not saved yet",
          node_count: 1,
        }}
      />,
    );

    expect(screen.getByText("Draft agent")).toBeDefined();
    expect(screen.getByText("Not saved yet")).toBeDefined();
    expect(screen.getByText("1 block")).toBeDefined();
  });

  it("pluralizes the block count and falls back to a default name", () => {
    render(<AgentPreviewCard output={{ node_count: 4 }} />);

    expect(screen.getByText("Agent preview")).toBeDefined();
    expect(screen.getByText("4 blocks")).toBeDefined();
  });
});

describe("SubSessionCard", () => {
  afterEach(cleanup);

  it("shows status, response, link and minute-scale elapsed time", () => {
    render(
      <SubSessionCard
        output={{
          status: "COMPLETED",
          response: "All finished",
          elapsed_seconds: 125,
          sub_autopilot_session_link: "/copilot?session=sub-1",
        }}
      />,
    );

    expect(screen.getByText("Sub-AutoPilot")).toBeDefined();
    expect(screen.getByText("completed")).toBeDefined();
    expect(screen.getByText("All finished")).toBeDefined();
    expect(screen.getByText("2m 5s")).toBeDefined();
    expect(screen.getByLabelText("Open sub-session").getAttribute("href")).toBe(
      "/copilot?session=sub-1",
    );
  });

  it("shows second-scale elapsed time without a link", () => {
    render(
      <SubSessionCard output={{ status: "RUNNING", elapsed_seconds: 45 }} />,
    );

    expect(screen.getByText("running")).toBeDefined();
    expect(screen.getByText("45s")).toBeDefined();
    expect(screen.queryByLabelText("Open sub-session")).toBeNull();
  });
});
