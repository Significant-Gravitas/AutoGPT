import { TestBed } from "@angular/core/testing";
import { vi } from "vitest";

import { App } from "./app";

describe("Angular Portside Cloud host", () => {
  beforeEach(async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
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
});
