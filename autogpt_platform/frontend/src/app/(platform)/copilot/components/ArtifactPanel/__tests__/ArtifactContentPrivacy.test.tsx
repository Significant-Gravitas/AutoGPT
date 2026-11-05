import { act, render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { ArtifactContent } from "../components/ArtifactContent";
import {
  clearContentCache,
  getCachedArtifactContent,
} from "../components/useArtifactContent";
import type { ArtifactRef } from "../../../store";
import { classifyArtifactRef } from "../helpers";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const originalUser = useAuthStore.getState().user;
const artifact: ArtifactRef = {
  id: "private-view",
  title: "Private.txt",
  mimeType: "text/plain",
  sourceUrl: "/api/proxy/api/workspace/files/private-view/download",
  origin: "agent",
};
const classification = classifyArtifactRef(artifact);
const privateSource = "Private source from first owner";

beforeEach(() => {
  clearContentCache();
  useAuthStore.setState({
    user: { id: "owner-a", email: "a@example.com", user_metadata: {} },
  });
});
afterEach(() => {
  clearContentCache();
  useAuthStore.setState({ user: originalUser });
});

describe("artifact source cache isolation", () => {
  it("does not display cached source when a new owner's artifact read is denied", async () => {
    let denied = false;
    let reads = 0;
    server.use(
      http.get("/api/proxy/api/workspace/files/private-view/download", () => {
        reads++;
        return denied
          ? HttpResponse.json({ detail: "Not found" }, { status: 404 })
          : HttpResponse.text(privateSource);
      }),
    );
    render(
      <ArtifactContent
        artifact={artifact}
        classification={classification}
        isSourceView={false}
      />,
    );
    expect(await screen.findByText(privateSource)).toBeDefined();
    expect(getCachedArtifactContent(artifact)).toBe(privateSource);
    const initialReads = reads;
    denied = true;
    act(() =>
      useAuthStore.setState({
        user: { id: "owner-b", email: "b@example.com", user_metadata: {} },
      }),
    );
    expect(screen.queryByText(privateSource)).toBeNull();
    expect(await screen.findByText("Failed to load content")).toBeDefined();
    expect(screen.queryByText(privateSource)).toBeNull();
    expect(getCachedArtifactContent(artifact)).toBeUndefined();
    expect(reads).toBeGreaterThan(initialReads);
  });

  it.each([undefined, "shared-document"])(
    "does not reuse a file under another saved organization with document %s",
    async (documentId) => {
      const savedA = {
        ...artifact,
        documentId,
        organizationId: "saved-a",
        teamId: null,
      };
      const savedB = {
        ...artifact,
        documentId,
        organizationId: "saved-b",
        teamId: null,
      };
      let reads = 0;
      let denied = false;
      server.use(
        http.get("/api/proxy/api/workspace/files/private-view/download", () => {
          reads++;
          return denied
            ? HttpResponse.json({ detail: "Not found" }, { status: 404 })
            : HttpResponse.text(privateSource);
        }),
      );
      const ui = render(
        <ArtifactContent
          artifact={savedA}
          classification={classification}
          isSourceView
        />,
      );
      expect(await screen.findByText(privateSource)).toBeDefined();
      expect(getCachedArtifactContent(savedA)).toBe(privateSource);
      expect(getCachedArtifactContent(savedB)).toBeUndefined();
      expect(getCachedArtifactContent(artifact.id)).toBeUndefined();
      const initialReads = reads;
      denied = true;
      ui.rerender(
        <ArtifactContent
          artifact={savedB}
          classification={classification}
          isSourceView
        />,
      );
      expect(screen.queryByText(privateSource)).toBeNull();
      expect(await screen.findByText("Failed to load content")).toBeDefined();
      expect(reads).toBeGreaterThan(initialReads);
    },
  );
});
