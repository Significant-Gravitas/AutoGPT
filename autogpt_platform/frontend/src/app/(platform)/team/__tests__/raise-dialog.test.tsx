import {
  getCreateRaisedExpertMockHandler,
  getGetRaiseIntroMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { CreateRaisedExpertRequest } from "@/app/api/__generated__/models/createRaisedExpertRequest";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { RaiseExpertButton } from "../components/RaiseExpertButton/RaiseExpertButton";

const { pushMock, replaceMock, toastMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  replaceMock: vi.fn(),
  toastMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: replaceMock,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

const intro = {
  intro: "I'll keep your dependency update PRs reviewed and merged.",
  questions: [{ question: "Which repo?", options: ["One", "Two", "Three"] }],
  is_generated: true,
};

function raisedExpert(body: CreateRaisedExpertRequest): Expert {
  return {
    id: body.id ?? "missing",
    name: body.name,
    avatar_url: null,
    role: body.role ?? "",
    job_title: body.job_title,
    tagline: null,
    bio: null,
    skills: [],
    identity: `I'm ${body.name}.`,
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    weekly_budget: 500,
  };
}

beforeEach(() => {
  server.use(
    getGetRaiseIntroMockHandler(intro),
    // Without a photo the expert's avatar is generated in the background;
    // here generation is unavailable, so it keeps the default one.
    http.post("/api/proxy/api/experts/avatars/generations", () =>
      HttpResponse.json({ detail: "Not configured" }, { status: 503 }),
    ),
  );
});

afterEach(() => {
  pushMock.mockReset();
  replaceMock.mockReset();
  toastMock.mockReset();
});

async function fillAndCreate(name: string, role: string) {
  await userEvent.click(
    screen.getByRole("button", { name: "Create an Expert" }),
  );
  await userEvent.type(await screen.findByLabelText("Name"), name);
  await userEvent.type(screen.getByLabelText("Role"), role);
  await userEvent.click(screen.getByRole("button", { name: "Create" }));
}

describe("create expert dialog", () => {
  test("asks only for a name and a role and opens the expert before the server answers", async () => {
    let release: () => void = () => {};
    const serverAnswered = new Promise<void>((resolve) => {
      release = resolve;
    });
    let sent: CreateRaisedExpertRequest | null = null;
    server.use(
      getCreateRaisedExpertMockHandler(async ({ request }) => {
        sent = (await request.json()) as CreateRaisedExpertRequest;
        await serverAnswered;
        return { expert: raisedExpert(sent), failed_attachments: [] };
      }),
    );
    render(<RaiseExpertButton />);

    await fillAndCreate("Deps merger", "Release engineer");

    expect(pushMock).toHaveBeenCalledTimes(1);
    const [href] = pushMock.mock.calls[0] as [string];
    const id = href.match(/^\/team\/([0-9a-f-]{36})\?welcome=1$/)?.[1];
    expect(id).toBeDefined();
    expect(screen.queryByLabelText("Name")).toBeNull();

    await waitFor(() => expect(sent).not.toBeNull());
    expect(sent).toEqual({
      id,
      name: "Deps merger",
      role: "Developer",
      job_title: "Release engineer",
      color: "blue-300",
    });
    release();
  });

  test("does nothing until the expert has a name", async () => {
    render(<RaiseExpertButton />);

    await userEvent.click(
      screen.getByRole("button", { name: "Create an Expert" }),
    );
    await userEvent.type(await screen.findByLabelText("Role"), "Researcher");

    expect(
      screen.getByRole("button", { name: "Create" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(pushMock).not.toHaveBeenCalled();
  });

  test("sends the user back to the team with the reason when the create fails", async () => {
    server.use(
      http.post("/api/proxy/api/experts/raise", () =>
        HttpResponse.json(
          { detail: { code: "active_expert_limit", limit: 20 } },
          { status: 409 },
        ),
      ),
    );
    render(<RaiseExpertButton />);

    await fillAndCreate("Inbox zero", "Support specialist");

    expect(pushMock).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/team"));
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Your team is full" }),
    );
  });

  test("rejects a photo that is not an image it can store", async () => {
    render(<RaiseExpertButton />);
    await userEvent.click(
      screen.getByRole("button", { name: "Create an Expert" }),
    );

    await userEvent.upload(
      await screen.findByTestId("raise-expert-photo-input"),
      new File(["%PDF"], "cv.pdf", { type: "application/pdf" }),
      { applyAccept: false },
    );

    expect((await screen.findByRole("alert")).textContent).toBe(
      "Use a PNG, JPEG or WebP image.",
    );
  });
});
