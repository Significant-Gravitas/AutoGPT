import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { getPostV2UpdateUserProfileMockHandler200 } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { server } from "@/mocks/mock-server";
import { ProfileInfoForm } from "../ProfileInfoForm";

vi.mock("next/image", async () =>
  vi.importActual<typeof import("next/image")>("next/image"),
);
const uploadSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/direct-upload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/direct-upload")>();
  return { ...actual, uploadSubmissionMediaDirect: uploadSpy };
});

function profile(avatar_url: string): ProfileDetails {
  return {
    name: "Local user",
    username: "local-user",
    description: "Profile",
    links: [],
    avatar_url,
    is_featured: false,
  };
}

afterEach(() => vi.unstubAllEnvs());
beforeEach(() => uploadSpy.mockReset());

it.each([
  [
    "http://localhost:8006/api",
    "http://localhost:8006/api/store/media/user/images/avatar.png",
  ],
  [
    "https://appliance.example/_agpt/api",
    "https://appliance.example/_agpt/api/store/media/user/images/avatar.png",
  ],
  ["http://localhost:8006/api", "/api/store/media/user/images/avatar.png"],
])(
  "renders saved and newly uploaded local avatars from %s",
  async (backend, url) => {
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", backend);
    const uploadedURL = url.replace("avatar.png", "uploaded.png");
    uploadSpy.mockResolvedValueOnce(uploadedURL);
    server.use(
      getPostV2UpdateUserProfileMockHandler200(() => profile(uploadedURL)),
    );
    render(<ProfileInfoForm profile={profile(url)} />);

    const image = screen.getByRole("img", { name: "Profile" });
    expect(image.getAttribute("src")).toBe(url);
    expect(image.hasAttribute("srcset")).toBe(false);

    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    fireEvent.change(input, {
      target: {
        files: [new File(["image"], "avatar.png", { type: "image/png" })],
      },
    });
    await vi.waitFor(() =>
      expect(
        screen.getByRole("img", { name: "Profile" }).getAttribute("src"),
      ).toBe(uploadedURL),
    );
    const uploaded = screen.getByRole("img", { name: "Profile" });
    expect(uploaded.hasAttribute("srcset")).toBe(false);
  },
);

it.each([
  "https://storage.googleapis.com/media/avatar.png",
  "https://untrusted.example/api/store/media/user/images/avatar.png",
])("preserves existing optimization for %s", (url) => {
  vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "http://localhost:8006/api");
  render(<ProfileInfoForm profile={profile(url)} />);
  const image = screen.getByRole("img", { name: "Profile" });
  expect(image.getAttribute("src")).toMatch(/^\/_next\/image\?/);
  expect(image.getAttribute("srcset")).toContain("/_next/image?");
});
