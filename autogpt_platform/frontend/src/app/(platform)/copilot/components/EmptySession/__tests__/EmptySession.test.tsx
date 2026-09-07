import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  normalizeWhitespace,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";

import { EmptySession } from "../EmptySession";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "hire-experts",
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

function makeExpert(args: { id: string; name: string; role: string }): Expert {
  return {
    ...args,
    avatar_url: `https://example.com/${args.id}.png`,
    bio: null,
    skills: [],
    tagline: "",
    identity: `You are ${args.name}.`,
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: `template-${args.id}`,
    is_archived: false,
    workflows: [],
  };
}

const mariaExpert = makeExpert({
  id: "expert-maria",
  name: "Maria",
  role: "Marketing Strategist",
});
const maxExpert = makeExpert({ id: "expert-max", name: "Max", role: "" });

function renderEmptySession(searchParams: string) {
  server.use(getListExpertIdentitiesMockHandler([mariaExpert, maxExpert]));
  const Wrapper = withNuqsTestingAdapter({ searchParams });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={() => {}}
      />
    </Wrapper>,
  );
}

describe("EmptySession — recipient-aware intro", () => {
  it("introduces the selected expert by name and role", async () => {
    const { container } = renderEmptySession("?expertId=expert-maria");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "I'm Maria, your Marketing Strategist. What should I take on?",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Maria work on?"),
    ).toBeDefined();
  });

  it("drops the role clause when the expert has none", async () => {
    const { container } = renderEmptySession("?expertId=expert-max");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "I'm Max. What should I take on?",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Max work on?"),
    ).toBeDefined();
  });

  it("keeps the Autopilot intro without a selected expert", async () => {
    const { container } = renderEmptySession("");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell me about your work — I'll find what to automate.",
      ),
    );
    expect(screen.getByPlaceholderText(/What's your role/)).toBeDefined();
  });
});
