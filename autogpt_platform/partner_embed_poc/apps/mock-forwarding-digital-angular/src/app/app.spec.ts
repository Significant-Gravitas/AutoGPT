import { TestBed } from "@angular/core/testing";
import { vi } from "vitest";

import { App, assistantNoticeFor, documentsForJobs } from "./app";

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
});
