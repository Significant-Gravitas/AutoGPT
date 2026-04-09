import { describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { ProfileInfoForm } from "../ProfileInfoForm";

function makeProfile(overrides: Partial<ProfileDetails> = {}): ProfileDetails {
  return {
    name: "Initial Name",
    username: "initial-user",
    description: "Initial description",
    links: [],
    avatar_url: "",
    ...overrides,
  } as ProfileDetails;
}

describe("ProfileInfoForm", () => {
  it("renders the existing profile values into editable fields", () => {
    render(<ProfileInfoForm profile={makeProfile({ name: "Hello World" })} />);
    const nameInput = screen.getByTestId(
      "profile-info-form-display-name",
    ) as HTMLInputElement;
    expect(nameInput.defaultValue).toBe("Hello World");
  });

  it("submits the new display name to POST /api/store/profile and reflects the response", async () => {
    let receivedBody: Record<string, unknown> | null = null;

    server.use(
      http.post(
        "http://localhost:3000/api/proxy/api/store/profile",
        async ({ request }) => {
          receivedBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            ...makeProfile({ name: receivedBody?.name as string }),
          });
        },
      ),
    );

    render(<ProfileInfoForm profile={makeProfile({ name: "Old Name" })} />);

    const nameInput = screen.getByTestId("profile-info-form-display-name");
    fireEvent.change(nameInput, { target: { value: "Brand New Name" } });

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    // Wait for the request to fire
    await new Promise((r) => setTimeout(r, 50));
    // Poll briefly until MSW handler captured the body
    for (let i = 0; i < 20 && receivedBody === null; i++) {
      await new Promise((r) => setTimeout(r, 50));
    }

    expect(
      receivedBody,
      "POST /api/store/profile must fire when the user clicks Save",
    ).not.toBeNull();
    expect(receivedBody!.name).toBe("Brand New Name");
  });

  it("does not silently swallow the request when the API returns 422", async () => {
    let calls = 0;
    server.use(
      http.post("http://localhost:3000/api/proxy/api/store/profile", () => {
        calls += 1;
        return HttpResponse.json(
          { detail: "validation error" },
          { status: 422 },
        );
      }),
    );

    render(<ProfileInfoForm profile={makeProfile()} />);

    const nameInput = screen.getByTestId("profile-info-form-display-name");
    fireEvent.change(nameInput, { target: { value: "Anything" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    for (let i = 0; i < 20 && calls === 0; i++) {
      await new Promise((r) => setTimeout(r, 50));
    }

    expect(
      calls,
      "save click must hit the backend even when validation fails",
    ).toBeGreaterThan(0);
  });
});
