import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, expect, test, vi } from "vitest";
import { HireOfficeGallery } from "../HireOfficeGallery";
import type { HireOfficeResponse, OfficeTemplate } from "../api";

// Strip Radix portals — happy-dom doesn't render them. The mock keeps the
// Dialog tree visible while ignoring controlled/forceOpen props.
function MockDialog({ children }: { children: React.ReactNode }) {
  return <div role="dialog">{children}</div>;
}
function MockDialogContent({ children }: { children: React.ReactNode }) {
  return <div>{children}</div>;
}
MockDialog.Content = MockDialogContent;
vi.mock("@/components/molecules/Dialog/Dialog", () => ({
  Dialog: MockDialog,
}));

const templates: OfficeTemplate[] = [
  {
    id: "office-growth",
    name: "Growth Office",
    description: "Marketing and outreach experts that grow your audience.",
    experts: [
      {
        template_id: "template-maria",
        name: "Maria",
        role: "Marketing Strategist",
        avatar_url: null,
        tagline: "Grows your brand while you sleep",
        schedule_cron: "0 9 * * 1",
        intro_task_title: "Audit your current marketing channels",
      },
      {
        template_id: "template-leo",
        name: "Leo",
        role: "Outreach Specialist",
        avatar_url: null,
        tagline: null,
        schedule_cron: null,
        intro_task_title: null,
      },
    ],
  },
  {
    id: "office-support",
    name: "Support Office",
    description: "Keeps your customers answered around the clock.",
    experts: [
      {
        template_id: "template-sam",
        name: "Sam",
        role: "Support Lead",
        avatar_url: null,
        tagline: null,
        schedule_cron: null,
        intro_task_title: null,
      },
    ],
  },
  {
    id: "office-ops",
    name: "Ops Office",
    description: "Back-office experts that keep operations humming.",
    experts: [
      {
        template_id: "template-ada",
        name: "Ada",
        role: "Operations Manager",
        avatar_url: null,
        tagline: null,
        schedule_cron: null,
        intro_task_title: null,
      },
    ],
  },
];

const hiredExpert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  tagline: "Grows your brand while you sleep",
  bio: null,
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
} as unknown as Expert;

const hireResponse: HireOfficeResponse = {
  office_template_id: "office-growth",
  office_name: "Growth Office",
  hired: [
    {
      expert: hiredExpert,
      intro_task_id: "task-1",
      intro_task_title: "Audit your current marketing channels",
      schedule_created: true,
    },
  ],
};

beforeEach(() => {
  server.use(
    http.get("/api/proxy/api/experts/office-templates", () =>
      HttpResponse.json(templates),
    ),
    http.post("/api/proxy/api/experts/hire-office", () =>
      HttpResponse.json(hireResponse),
    ),
  );
});

test("renders the three office packs with expert counts", async () => {
  render(<HireOfficeGallery />);

  expect(await screen.findByText("Growth Office")).toBeTruthy();
  expect(screen.getByText("Support Office")).toBeTruthy();
  expect(screen.getByText("Ops Office")).toBeTruthy();
  expect(screen.getByText("2 experts")).toBeTruthy();
});

test("clicking a pack previews its experts", async () => {
  const user = userEvent.setup();
  render(<HireOfficeGallery />);

  await user.click(
    await screen.findByRole("button", { name: /Growth Office/ }),
  );

  const list = await screen.findByRole("list", {
    name: "Experts in this office",
  });
  expect(list).toBeTruthy();
  expect(screen.getByText("Maria")).toBeTruthy();
  expect(screen.getByText("Marketing Strategist")).toBeTruthy();
  expect(screen.getByText("Grows your brand while you sleep")).toBeTruthy();
  expect(screen.getByText("Scheduled")).toBeTruthy();
  expect(
    screen.getByText(/First task: Audit your current marketing channels/),
  ).toBeTruthy();
  expect(screen.getByRole("button", { name: "Hire office" })).toBeTruthy();
});

test("hiring shows the roster success state", async () => {
  const user = userEvent.setup();
  render(<HireOfficeGallery />);

  await user.click(
    await screen.findByRole("button", { name: /Growth Office/ }),
  );
  await user.click(screen.getByRole("button", { name: "Hire office" }));

  expect(await screen.findByText(/1 expert joined your team/)).toBeTruthy();
  const roster = screen.getByRole("list", { name: "Hired experts" });
  expect(roster).toBeTruthy();
  expect(
    screen.getByText("Audit your current marketing channels"),
  ).toBeTruthy();
  expect(screen.getByText("Queued")).toBeTruthy();
  expect(screen.getByText("Schedule created")).toBeTruthy();
});
