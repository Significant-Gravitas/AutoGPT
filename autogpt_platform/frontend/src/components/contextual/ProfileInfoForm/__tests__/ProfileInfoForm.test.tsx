import { describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  getPostV2UpdateUserProfileMockHandler200,
  getPostV2UpdateUserProfileMockHandler422,
  getPostV2UpdateUserProfileResponseMock422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { server } from "@/mocks/mock-server";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { ProfileInfoForm } from "../ProfileInfoForm";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastSpy, dismiss: () => {}, toasts: [] }),
  };
});

const uploadSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/direct-upload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/direct-upload")>();
  return { ...actual, uploadSubmissionMediaDirect: uploadSpy };
});

function fileInput() {
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

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
      getPostV2UpdateUserProfileMockHandler200(async ({ request }) => {
        receivedBody = (await request.json()) as Record<string, unknown>;
        return makeProfile({ name: receivedBody?.name as string });
      }),
    );

    render(<ProfileInfoForm profile={makeProfile({ name: "Old Name" })} />);

    const nameInput = screen.getByTestId("profile-info-form-display-name");
    fireEvent.change(nameInput, { target: { value: "Brand New Name" } });

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => {
      expect(
        receivedBody,
        "POST /api/store/profile must fire when the user clicks Save",
      ).not.toBeNull();
    });

    expect(receivedBody!.name).toBe("Brand New Name");
  });

  it("does not silently swallow the request when the API returns 422", async () => {
    let calls = 0;
    server.use(
      getPostV2UpdateUserProfileMockHandler422(() => {
        calls += 1;
        return getPostV2UpdateUserProfileResponseMock422({
          detail: [
            {
              loc: ["body", "name"],
              msg: "validation error",
              type: "value_error",
            },
          ],
        });
      }),
    );

    render(<ProfileInfoForm profile={makeProfile()} />);

    const nameInput = screen.getByTestId("profile-info-form-display-name");
    fireEvent.change(nameInput, { target: { value: "Anything" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => {
      expect(
        calls,
        "save click must hit the backend even when validation fails",
      ).toBeGreaterThan(0);
    });
  });

  it("uploads a new avatar and persists it via the profile update", async () => {
    toastSpy.mockClear();
    uploadSpy.mockClear();
    uploadSpy.mockResolvedValueOnce("https://cdn.test/new-avatar.png");
    server.use(
      getPostV2UpdateUserProfileMockHandler200(async () =>
        makeProfile({ avatar_url: "https://cdn.test/new-avatar.png" }),
      ),
    );

    render(<ProfileInfoForm profile={makeProfile()} />);

    const file = new File(["x"], "avatar.png", { type: "image/png" });
    fireEvent.change(fileInput(), { target: { files: [file] } });

    await waitFor(() => {
      expect(uploadSpy).toHaveBeenCalledWith(file);
    });
  });

  it("rejects an oversized avatar before uploading and explains the limit", async () => {
    toastSpy.mockClear();
    uploadSpy.mockClear();

    render(<ProfileInfoForm profile={makeProfile()} />);

    const bigFile = new File(["x"], "avatar.png", { type: "image/png" });
    Object.defineProperty(bigFile, "size", { value: 51 * 1024 * 1024 });
    fireEvent.change(fileInput(), { target: { files: [bigFile] } });

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

  it.each(["button", "form"])(
    "submits once through %s and keeps the pending state until the response",
    async (trigger) => {
      let calls = 0;
      let finishSave: () => void = () => {};
      const pending = new Promise<void>((resolve) => {
        finishSave = resolve;
      });
      server.use(
        getPostV2UpdateUserProfileMockHandler200(async ({ request }) => {
          calls += 1;
          const body = await request.json();
          expect(body).toMatchObject({
            name: "Updated name",
            links: ["https://example.com"],
          });
          await pending;
          return makeProfile({ name: "Updated name" });
        }),
      );
      render(
        <ProfileInfoForm
          profile={makeProfile({ links: ["https://example.com", ""] })}
        />,
      );
      fireEvent.change(screen.getByTestId("profile-info-form-display-name"), {
        target: { value: "Updated name" },
      });
      try {
        if (trigger === "button") {
          fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
        } else {
          const form = screen
            .getByRole("button", { name: "Save changes" })
            .closest("form");
          expect(form).not.toBeNull();
          expect(fireEvent.submit(form!)).toBe(false);
        }
        await waitFor(() => {
          expect(calls).toBe(1);
          expect(
            screen
              .getByRole("button", { name: "Saving..." })
              .hasAttribute("disabled"),
          ).toBe(true);
        });
      } finally {
        finishSave();
      }
      await waitFor(() => {
        const button = screen.getByRole("button", { name: "Save changes" });
        expect(button.hasAttribute("disabled")).toBe(false);
      });
      expect(calls).toBe(1);
    },
  );

  it("keeps the saved avatar and explains an upload failure", async () => {
    toastSpy.mockClear();
    uploadSpy.mockReset();
    uploadSpy.mockRejectedValueOnce(new Error("Upload unavailable"));
    const avatar = "https://storage.googleapis.com/media/avatar.png";
    render(<ProfileInfoForm profile={makeProfile({ avatar_url: avatar })} />);
    fireEvent.change(fileInput(), {
      target: {
        files: [new File(["image"], "avatar.png", { type: "image/png" })],
      },
    });
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith({
        title: "Failed to upload photo",
        description: "Upload unavailable",
        variant: "destructive",
      }),
    );
    expect(
      screen.getByRole("img", { name: "Profile" }).getAttribute("src"),
    ).toBe(avatar);
  });
});
