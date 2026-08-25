import { TestBed } from "@angular/core/testing";
import { vi } from "vitest";

import {
  App,
  agentPermissionMessageFor,
  assistantNoticeFor,
  documentsForJobs,
  suggestedPromptsFor,
} from "./app";

describe("Angular Portside Cloud host", () => {
  beforeEach(async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ required: false, authorized: true }), {
            status: 200,
          }),
        )
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({
              users: [
                {
                  id: "fd-user-1042",
                  email: "alex@example.com",
                  name: "Alex Morgan",
                  organizations: ["fd-account-77", "fd-account-88"],
                },
              ],
            }),
            { status: 200 },
          ),
        )
        .mockResolvedValueOnce(new Response(null, { status: 401 })),
    );
    await TestBed.configureTestingModule({
      imports: [App],
    }).compileComponents();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the partner-owned sign-in before AutoGPT is involved", async () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    await vi.waitFor(() => {
      fixture.detectChanges();
      expect(fixture.nativeElement.textContent).toContain("Alex Morgan");
    });
    expect(fixture.nativeElement.textContent).toContain(
      "embedded assistant never asks users to create a second account",
    );
  });

  it("describes saved agents without exposing their internal ID", () => {
    expect(assistantNoticeFor("/library/agents/agent-123")).toBe(
      "Saved agent is ready in the automation library.",
    );
  });

  it("builds document rows only from the active tenant's jobs", () => {
    const documents = documentsForJobs([
      { reference: "HBR-2208", status: "Rail slot pending" },
      { reference: "HBR-2231", status: "On schedule" },
    ]);

    expect(documents.map((document) => document.name)).toEqual([
      "Arrival notice · HBR-2208",
      "Bill of lading · HBR-2231",
    ]);
  });

  it("offers restricted operators only capability-safe prompts", () => {
    const capabilities = ["jobs.read", "documents.read"];

    expect(suggestedPromptsFor("documents", capabilities)).toEqual([
      "Find jobs with missing documents and produce an exception summary.",
    ]);
    expect(suggestedPromptsFor("automations", capabilities)).toEqual([
      "Turn the current shipment exceptions into a repeatable checklist a manager could automate.",
      "Review this tenant's document gaps and outline the safest manual follow-up workflow.",
    ]);
  });

  it("offers the full agent lifecycle only with matching capabilities", () => {
    const prompts = suggestedPromptsFor("automations", [
      "agents.create",
      "agents.run",
      "agents.schedule",
      "autogpt:block:calculator",
    ]);

    expect(prompts).toHaveLength(3);
    expect(prompts[0]).toContain("Create and save");
    expect(prompts[1]).toContain("Run the saved");
    expect(prompts[2]).toContain("schedule the saved");
  });

  it("describes partial agent controls without overclaiming access", () => {
    expect(agentPermissionMessageFor(["agents.run"])).toBe(
      "Agent controls enabled for this role: run. Other actions remain unavailable.",
    );
    expect(
      agentPermissionMessageFor([
        "agents.create",
        "agents.run",
        "agents.schedule",
      ]),
    ).toBe(
      "Agent controls enabled for this role: run and schedule. Other actions remain unavailable.",
    );
  });
});
