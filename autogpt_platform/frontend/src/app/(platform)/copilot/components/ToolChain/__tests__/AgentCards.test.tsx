import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
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

  it("renders the expert avatar url through the shared avatar", () => {
    // Only asserts the card hands the backend url to ExpertAvatar unchanged.
    // ExpertAvatar still renders via next/image, so the configured-hostname
    // allow list is not covered here — setup-nextjs-mocks swaps in a plain img.
    render(
      <SubSessionCard
        output={{
          status: "running",
          expert: {
            id: "exp-1",
            name: "Maria",
            role: "Researcher",
            avatar_url: "https://cdn.example.com/maria.png",
          },
        }}
      />,
    );

    expect(screen.getByAltText("Maria").getAttribute("src")).toBe(
      "https://cdn.example.com/maria.png",
    );
    expect(screen.getByText("Researcher")).toBeDefined();
  });

  it("shows second-scale elapsed time without a link", () => {
    render(
      <SubSessionCard output={{ status: "RUNNING", elapsed_seconds: 45 }} />,
    );

    expect(screen.getByText("running")).toBeDefined();
    expect(screen.getByText("45s")).toBeDefined();
    expect(screen.queryByLabelText("Open sub-session")).toBeNull();
  });

  // A running delegation used to show only a ticking clock. It now reads as a
  // tracked job: an up-front estimate, the elapsed time, and which phase of
  // the delegate's own plan the work has reached.
  it("shows the estimate, the elapsed time and the phase timeline together", () => {
    render(
      <SubSessionCard
        output={{
          status: "RUNNING",
          elapsed_seconds: 125,
          estimated_minutes: 25,
          sub_autopilot_session_link: "/copilot?sessionId=sub-1",
          phases: [
            { content: "Scaffold the agent", status: "completed" },
            { content: "Wire the integrations", status: "in_progress" },
            { content: "Run a smoke test", status: "pending" },
          ],
        }}
      />,
    );

    expect(screen.getByText("~25m est")).toBeDefined();
    expect(screen.getByText("2m 5s")).toBeDefined();
    expect(screen.getByText("1 of 3 steps done")).toBeDefined();
    expect(screen.getByText("Scaffold the agent")).toBeDefined();
    expect(screen.getByText("Wire the integrations")).toBeDefined();
    expect(screen.getByText("Run a smoke test")).toBeDefined();
    expect(screen.getByLabelText("completed")).toBeDefined();
    expect(screen.getByLabelText("in progress")).toBeDefined();
    expect(screen.getByLabelText("pending")).toBeDefined();
    // The deep link survives the extra chrome.
    expect(screen.getByLabelText("Open sub-session").getAttribute("href")).toBe(
      "/copilot?sessionId=sub-1",
    );
  });

  it("renders no estimate chip and no timeline when the backend sent neither", () => {
    render(
      <SubSessionCard output={{ status: "RUNNING", elapsed_seconds: 45 }} />,
    );

    expect(screen.queryByText(/est$/)).toBeNull();
    expect(screen.queryByText(/steps done$/)).toBeNull();
  });

  it("ignores a non-positive estimate instead of promising 0 minutes", () => {
    render(
      <SubSessionCard
        output={{ status: "RUNNING", estimated_minutes: 0, elapsed_seconds: 5 }}
      />,
    );

    expect(screen.queryByText(/est$/)).toBeNull();
  });

  it("drops malformed phase entries rather than rendering blank rows", () => {
    render(
      <SubSessionCard
        output={{
          status: "RUNNING",
          phases: [
            { content: "Real step", status: "completed" },
            { content: "   ", status: "pending" },
            { status: "pending" },
            "junk",
            null,
          ],
        }}
      />,
    );

    expect(screen.getByText("1 of 1 steps done")).toBeDefined();
    expect(screen.getByText("Real step")).toBeDefined();
  });

  it("treats an unknown phase status as pending", () => {
    render(
      <SubSessionCard
        output={{
          status: "RUNNING",
          phases: [{ content: "Mystery step", status: "wat" }],
        }}
      />,
    );

    expect(screen.getByText("Mystery step")).toBeDefined();
    expect(screen.getByLabelText("pending")).toBeDefined();
  });
});
