import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import type { ExpertArtifact } from "../../../../store";
import { ExpertArtifactContent } from "../ExpertArtifactContent";

const fetched = {
  enabled: false,
  record: {
    identity: "You ship the frontend.",
    boundaries: "You never touch the backend.",
    voice_preferences: "Crisp.",
    weekly_budget: 500,
    role: "Frontend Engineer",
    tagline: null,
    avatar_url: null,
    color: "",
  },
};
vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  useGetExpert: (
    _id: string,
    options: { query: { enabled: boolean; select: (r: unknown) => unknown } },
  ) => {
    fetched.enabled = options.query.enabled;
    return {
      data: options.query.enabled
        ? options.query.select({ status: 200, data: fetched.record })
        : undefined,
    };
  },
}));

function expert(overrides: Partial<ExpertArtifact> = {}): ExpertArtifact {
  return {
    id: null,
    kind: "raise",
    name: "Otto",
    role: "Inbox triage",
    color: "violet-300",
    tagline: "Sorts your morning inbox.",
    about: "You group the morning inbox.",
    boundaries: "You never send a reply yourself.",
    voicePreferences: "Plain sentences.",
    weeklyBudget: 2000,
    avatarUrl: null,
    applied: false,
    ...overrides,
  };
}

describe("ExpertArtifactContent", () => {
  it("lays out every detail the proposal carries", () => {
    render(<ExpertArtifactContent expert={expert()} />);

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Inbox triage")).toBeDefined();
    expect(screen.getByText("Sorts your morning inbox.")).toBeDefined();
    expect(screen.getByText(/nothing created yet/i)).toBeDefined();
    expect(screen.getByText("Raised from a charter")).toBeDefined();
    expect(screen.getByText("You group the morning inbox.")).toBeDefined();
    expect(screen.getByText("You never send a reply yourself.")).toBeDefined();
    expect(screen.getByText("Plain sentences.")).toBeDefined();
    expect(screen.getByText("2000 credits")).toBeDefined();
    expect(screen.getByText("violet-300")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("reads a default budget and the team status once applied", () => {
    render(
      <ExpertArtifactContent
        expert={expert({ id: "exp-1", applied: true, weeklyBudget: null })}
      />,
    );

    expect(screen.getByText("On the team")).toBeDefined();
    expect(screen.getByText("Default")).toBeDefined();
    expect(screen.getByText("exp-1")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("fills a saved applied card's missing charter from the expert record", () => {
    render(
      <ExpertArtifactContent
        expert={expert({
          id: "exp-1",
          applied: true,
          about: null,
          boundaries: null,
          voicePreferences: null,
          weeklyBudget: null,
        })}
      />,
    );

    expect(fetched.enabled).toBe(true);
    expect(screen.getByText("You ship the frontend.")).toBeDefined();
    expect(screen.getByText("You never touch the backend.")).toBeDefined();
    expect(screen.getByText("Crisp.")).toBeDefined();
    expect(screen.getByText("500 credits")).toBeDefined();
  });

  it("does not fetch when the charter is already there", () => {
    render(<ExpertArtifactContent expert={expert({ id: "exp-1" })} />);

    expect(fetched.enabled).toBe(false);
    expect(screen.getByText("You group the morning inbox.")).toBeDefined();
  });
});
