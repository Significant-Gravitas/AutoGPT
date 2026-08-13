import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { InstallOnExpertButton } from "../InstallOnExpertButton";

const flagState = vi.hoisted(() => ({ hireExperts: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.HIRE_EXPERTS
        ? flagState.hireExperts
        : actual.useGetFlag(flag as never),
  };
});

// Signed-in on purpose: flag-off must hide the action and skip the experts
// request even for a logged-in user with hired experts.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true }),
}));

vi.mock(
  "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker",
  () => ({
    InstallWorkflowPicker: () => null,
  }),
);

const hiredExpert: Expert = {
  id: "expert-1",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: null,
  identity: "You are Maria.",
  voice_preferences: "Warm and direct.",
  boundaries: "Ask before external actions.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

describe("InstallOnExpertButton", () => {
  beforeEach(() => {
    flagState.hireExperts = false;
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("flag-off: renders no action and fires no experts request for a signed-in user with hired experts", async () => {
    const expertsSpy = vi.fn(() => [hiredExpert]);
    server.use(getListExpertsMockHandler(expertsSpy));

    render(<InstallOnExpertButton storeListingVersionId="slv-1" />);

    // Poll for a full window (same pipeline latency the flag-on test needs)
    // so a wrongly-enabled query + rendered action cannot slip past a
    // too-early assertion; findByRole must time out for the gate to hold.
    await expect(
      screen.findByRole(
        "button",
        { name: /install on expert/i },
        { timeout: 1000 },
      ),
    ).rejects.toThrow();
    expect(expertsSpy).not.toHaveBeenCalled();
  });

  test("flag-on: fetches experts and renders the action for the same signed-in user", async () => {
    flagState.hireExperts = true;
    const expertsSpy = vi.fn(() => [hiredExpert]);
    server.use(getListExpertsMockHandler(expertsSpy));

    render(<InstallOnExpertButton storeListingVersionId="slv-1" />);

    expect(
      await screen.findByRole("button", { name: /install on expert/i }),
    ).toBeDefined();
    expect(expertsSpy).toHaveBeenCalled();
  });
});
