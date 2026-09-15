import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

// With the flag off the recommendation endpoint 404s and is never called,
// so nothing will ever settle the section on its own — the panel has to
// treat "flag off" as an answer or it shows an empty list forever.
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

import { ConnectToolsPanel } from "../ConnectToolsPanel";

const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const RECOMMENDED_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-providers";

describe("ConnectToolsPanel — brain dump flag off", () => {
  it("shows the popular providers without asking for recommendations", async () => {
    let recommendationRequests = 0;
    server.use(
      http.get(PROVIDERS_URL, () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and wikis",
            supported_auth_types: ["oauth2"],
          },
          {
            name: "slack",
            description: "Team chat",
            supported_auth_types: ["oauth2"],
          },
        ]),
      ),
      http.get(CREDENTIALS_URL, () => HttpResponse.json([])),
      http.get(RECOMMENDED_URL, () => {
        recommendationRequests += 1;
        return HttpResponse.json({ ready: true, providers: [] });
      }),
    );

    render(<ConnectToolsPanel onBack={vi.fn()} onNext={vi.fn()} />);

    expect(await screen.findByText("Popular places to start")).toBeDefined();
    expect(screen.getByRole("button", { name: /Notion/ })).toBeDefined();
    expect(recommendationRequests).toBe(0);
  });
});
