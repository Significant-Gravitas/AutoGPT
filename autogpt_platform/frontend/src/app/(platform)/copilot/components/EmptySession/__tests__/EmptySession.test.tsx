import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
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
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EmptySession } from "../EmptySession";

const flags = vi.hoisted(() => ({ experts: true }));

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
    useGetFlag: (flag: string) => flag === "hire-experts" && flags.experts,
    useFlagStatus: (flag: string) => ({
      enabled: flag === "hire-experts" && flags.experts,
      ready: true,
    }),
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
const samExpert = makeExpert({ id: "expert-sam", name: "Sam", role: "Sales" });

function renderEmptySession(searchParams: string, onSend = vi.fn()) {
  server.use(
    getListExpertIdentitiesMockHandler([mariaExpert, maxExpert, samExpert]),
  );
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={onSend}
      />
    </Wrapper>,
  );
}

describe("EmptySession — recipient-aware intro", () => {
  it("introduces the selected expert with an inline recipient chip", async () => {
    const { container } = renderEmptySession("?expertId=expert-maria");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell Maria what you need, and it will get to work.",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Maria work on?"),
    ).toBeDefined();
  });

  it("supports an expert with a short role", async () => {
    const { container } = renderEmptySession("?expertId=expert-sam");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell Sam what you need, and it will get to work.",
      ),
    );
  });

  it("supports an expert without a role", async () => {
    const { container } = renderEmptySession("?expertId=expert-max");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell Max what you need, and it will get to work.",
      ),
    );
    expect(
      screen.getByPlaceholderText("What should Max work on?"),
    ).toBeDefined();
  });

  it("keeps the Otto intro without a selected expert", async () => {
    const { container } = renderEmptySession("");

    await waitFor(() =>
      expect(normalizeWhitespace(container)).toContain(
        "Tell Otto about your work, and it will find what to automate.",
      ),
    );
    expect(screen.getByPlaceholderText(/What's your role/)).toBeDefined();
  });
});

beforeEach(() => {
  flags.experts = true;
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
  );
});

it("shows the task prompt above the home recap", async () => {
  renderEmptySession("");
  const prompt = screen.getByPlaceholderText(/What's your role/);
  const recap = await screen.findByRole("heading", { name: "Your recap" });
  expect(
    prompt.compareDocumentPosition(recap) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Recent work" })).toBeDefined();
  expect(screen.getByRole("heading", { name: "Your team" })).toBeDefined();
  expect(screen.getByRole("heading", { name: "Now & next" })).toBeDefined();
});

it("keeps the prompt usable when the recap fails to load", async () => {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json({ detail: "failed" }, { status: 500 }),
    ),
  );
  const onSend = vi.fn();
  const user = userEvent.setup();
  renderEmptySession("", onSend);
  await screen.findByText("Your Home briefing could not be loaded");
  await user.type(
    screen.getByPlaceholderText(/What's your role/),
    "Plan my day{Enter}",
  );
  await waitFor(() => expect(onSend).toHaveBeenCalled());
  expect(onSend.mock.calls[0][0]).toBe("Plan my day");
});

it("keeps new tasks available without requesting the recap when experts is off", async () => {
  flags.experts = false;
  const requestedPaths: string[] = [];
  function record({ request }: { request: Request }) {
    requestedPaths.push(new URL(request.url).pathname);
  }
  server.events.on("request:start", record);
  try {
    renderEmptySession("");
    expect(
      await screen.findByPlaceholderText(/What's your role/),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { name: "Your recap" })).toBeNull();
    expect(requestedPaths).not.toContain("/api/proxy/api/home");
  } finally {
    server.events.removeListener("request:start", record);
  }
});

it("shows a named kickoff status and withholds the empty composer", () => {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "" });
  render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={() => {}}
        isKickoffStarting
        expertName="Maria"
      />
    </Wrapper>,
  );
  expect(screen.getByRole("status").textContent).toContain(
    "Opening Maria's workspace",
  );
  expect(screen.queryByPlaceholderText(/What's your role/)).toBeNull();
});

it("switches recipients from the intro with no duplicate picker in the composer", async () => {
  const user = userEvent.setup();
  renderEmptySession("");
  const picker = await screen.findByRole("button", {
    name: "Sending to Otto — change recipient",
  });
  const prompt = screen.getByPlaceholderText(/What's your role/);
  expect(
    picker.compareDocumentPosition(prompt) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(
    screen.getAllByRole("button", { name: /change recipient/, hidden: true }),
  ).toHaveLength(1);
  await user.click(picker);
  await user.click(await screen.findByRole("menuitem", { name: /Maria/ }));
  expect(
    await screen.findByPlaceholderText("What should Maria work on?"),
  ).toBeDefined();
  expect(
    screen.getByRole("button", { name: "Sending to Maria — change recipient" }),
  ).toBeDefined();
});
