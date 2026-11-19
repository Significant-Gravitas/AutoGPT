import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import { ApiError } from "@/lib/autogpt-server-api/helpers";

import { OAuthAppsSection } from "../OAuthAppsSection";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: unknown[]) => toastSpy(...args),
    useToast: () => ({ toast: toastSpy, dismiss: () => {}, toasts: [] }),
  };
});

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => false,
}));

const uploadSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/direct-upload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/direct-upload")>();
  return { ...actual, uploadOAuthAppLogoDirect: uploadSpy };
});

const app = {
  id: "app-1",
  name: "My App",
  client_id: "client-abc",
  description: "An app",
  logo_url: null,
  redirect_uris: ["https://app.test/cb"],
  grant_types: ["authorization_code"],
  scopes: [],
  owner_id: "user-1",
  is_active: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
};

vi.mock("@/app/api/__generated__/endpoints/oauth/oauth", () => ({
  useGetOauthListMyOauthApps: () => ({ data: [app], isLoading: false }),
  usePatchOauthUpdateAppStatus: () => ({ mutateAsync: vi.fn() }),
  getGetOauthListMyOauthAppsQueryKey: () => ["oauth-apps"],
}));

function logoInput() {
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

describe("OAuthAppsSection - logo upload", () => {
  beforeEach(() => {
    toastSpy.mockClear();
    uploadSpy.mockClear();
  });

  it("uploads a logo and shows a success toast", async () => {
    uploadSpy.mockResolvedValueOnce(undefined);

    render(<OAuthAppsSection />);
    await screen.findByText("My App");

    const file = new File(["x"], "logo.png", { type: "image/png" });
    fireEvent.change(logoInput(), { target: { files: [file] } });

    await waitFor(() => {
      expect(uploadSpy).toHaveBeenCalledWith("app-1", file);
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Success" }),
      );
    });
  });

  it("rejects an oversized logo before uploading and explains the 3MB limit", async () => {
    render(<OAuthAppsSection />);
    await screen.findByText("My App");

    const bigFile = new File(["x"], "logo.png", { type: "image/png" });
    Object.defineProperty(bigFile, "size", { value: 4 * 1024 * 1024 });
    fireEvent.change(logoInput(), { target: { files: [bigFile] } });

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "File too large",
          variant: "destructive",
        }),
      );
    });
    expect(uploadSpy).not.toHaveBeenCalled();
  });

  it("sends an account without a plan to whoever runs this server, where nothing bills", async () => {
    uploadSpy.mockRejectedValueOnce(
      new ApiError("A subscription is required.", 402, null),
    );

    render(<OAuthAppsSection />);
    await screen.findByText("My App");
    const file = new File(["x"], "logo.png", { type: "image/png" });
    fireEvent.change(logoInput(), { target: { files: [file] } });

    await waitFor(() => expect(toastSpy).toHaveBeenCalled());
    const [paywall] = toastSpy.mock.calls[0];
    expect(paywall).toMatchObject({
      title: "Uploading a logo needs an active plan",
    });
    render(
      <>
        {paywall.description}
        {paywall.action}
      </>,
    );
    expect(
      screen.getByText("Ask whoever runs this server to set one up."),
    ).toBeDefined();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Error" }),
    );
  });

  it("surfaces a destructive toast when the upload fails", async () => {
    uploadSpy.mockRejectedValueOnce(new Error("boom"));

    render(<OAuthAppsSection />);
    await screen.findByText("My App");

    const file = new File(["x"], "logo.png", { type: "image/png" });
    fireEvent.change(logoInput(), { target: { files: [file] } });

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Error", variant: "destructive" }),
      );
    });
  });
});
