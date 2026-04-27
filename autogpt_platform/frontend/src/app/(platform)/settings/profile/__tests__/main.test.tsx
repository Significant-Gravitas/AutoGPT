import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetUserProfileMockHandler200,
  getGetV2GetUserProfileMockHandler401,
  getPostV2UpdateUserProfileMockHandler200,
  getPostV2UpdateUserProfileMockHandler422,
  getPostV2UploadSubmissionMediaMockHandler200,
  getPostV2UploadSubmissionMediaMockHandler401,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";

import SettingsProfilePage from "../page";

const mockUseSupabase = vi.hoisted(() => vi.fn());
const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

function makeProfile(overrides: Partial<ProfileDetails> = {}): ProfileDetails {
  return {
    username: "jane_doe",
    name: "Jane Doe",
    description: "I build agents",
    avatar_url: "https://cdn.example.com/jane.png",
    links: ["https://jane.dev", "https://github.com/jane"],
    is_featured: false,
    ...overrides,
  };
}

function authenticate() {
  mockUseSupabase.mockReturnValue({
    user: {
      id: "user-1",
      email: "user@example.com",
      app_metadata: {},
      user_metadata: {},
      aud: "authenticated",
      created_at: "2026-01-01T00:00:00.000Z",
    },
    isLoggedIn: true,
    isUserLoading: false,
    supabase: {},
  });
}

beforeEach(() => {
  toastSpy.mockClear();
  authenticate();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("SettingsProfilePage - data loading", () => {
  test("renders the page heading and description copy", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    expect(
      await screen.findByRole("heading", { name: /^profile$/i, level: 1 }),
    ).toBeDefined();
    expect(screen.getByText(/manage how you appear/i)).toBeDefined();
  });

  test("hydrates the form fields from the loaded profile", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    expect(
      ((await screen.findByLabelText(/display name/i)) as HTMLInputElement)
        .value,
    ).toBe("Jane Doe");
    expect((screen.getByLabelText(/^handle$/i) as HTMLInputElement).value).toBe(
      "jane_doe",
    );
    expect(
      (
        screen.getByPlaceholderText(
          /tell people what you build/i,
        ) as HTMLTextAreaElement | null
      )?.value,
    ).toBe("I build agents");
    expect((screen.getByLabelText(/^link 1$/i) as HTMLInputElement).value).toBe(
      "https://jane.dev",
    );
    expect((screen.getByLabelText(/^link 2$/i) as HTMLInputElement).value).toBe(
      "https://github.com/jane",
    );
  });

  test("renders an error card and recovers via Try again", async () => {
    server.use(getGetV2GetUserProfileMockHandler401());

    render(<SettingsProfilePage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
    expect(
      screen.queryByRole("heading", { name: /^profile$/i, level: 1 }),
    ).toBeNull();

    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(
      await screen.findByRole("heading", { name: /^profile$/i, level: 1 }),
    ).toBeDefined();
  });
});

describe("SettingsProfilePage - validation", () => {
  test("shows an error and disables Save when display name is cleared", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "" } });

    expect(await screen.findByText(/display name is required/i)).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: /save changes/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  test("rejects an invalid handle and accepts a valid one", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const handle = (await screen.findByLabelText(
      /^handle$/i,
    )) as HTMLInputElement;
    fireEvent.change(handle, { target: { value: "no spaces here" } });

    expect(await screen.findByText(/2.+30/)).toBeDefined();

    fireEvent.change(handle, { target: { value: "jane_smith" } });

    await waitFor(() => {
      expect(screen.queryByText(/2.+30/)).toBeNull();
    });
  });

  test("shows the live character counter for the bio", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ description: "x".repeat(100) }),
      ),
    );

    render(<SettingsProfilePage />);

    expect(await screen.findByText("180 left")).toBeDefined();
  });
});

describe("SettingsProfilePage - markdown toolbar & preview", () => {
  test("Bold button wraps selected text with **", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ description: "hello world" }),
      ),
    );

    render(<SettingsProfilePage />);

    const textarea = (await screen.findByPlaceholderText(
      /tell people what you build/i,
    )) as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(0, 5);

    fireEvent.click(screen.getByRole("button", { name: /^bold$/i }));

    await waitFor(() => {
      expect(textarea.value).toBe("**hello** world");
    });
  });

  test("Italic button without selection inserts placeholder italic text", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile({ description: "" })),
    );

    render(<SettingsProfilePage />);

    const textarea = (await screen.findByPlaceholderText(
      /tell people what you build/i,
    )) as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(0, 0);

    fireEvent.click(screen.getByRole("button", { name: /^italic$/i }));

    await waitFor(() => {
      expect(textarea.value).toBe("*italic text*");
    });
  });

  test("Bulleted list prepends '- ' at the start of the line, idempotently", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ description: "first item" }),
      ),
    );

    render(<SettingsProfilePage />);

    const textarea = (await screen.findByPlaceholderText(
      /tell people what you build/i,
    )) as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(0, 0);

    fireEvent.click(screen.getByRole("button", { name: /bulleted list/i }));

    await waitFor(() => {
      expect(textarea.value).toBe("- first item");
    });

    // Second click should not double-prefix
    fireEvent.click(screen.getByRole("button", { name: /bulleted list/i }));
    expect(textarea.value).toBe("- first item");
  });

  test("Strikethrough wraps selection with ~~", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile({ description: "abc" })),
    );

    render(<SettingsProfilePage />);

    const textarea = (await screen.findByPlaceholderText(
      /tell people what you build/i,
    )) as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(0, 3);

    fireEvent.click(screen.getByRole("button", { name: /strikethrough/i }));

    await waitFor(() => {
      expect(textarea.value).toBe("~~abc~~");
    });
  });

  test("Link button inserts the markdown link template at cursor", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile({ description: "" })),
    );

    render(<SettingsProfilePage />);

    const textarea = (await screen.findByPlaceholderText(
      /tell people what you build/i,
    )) as HTMLTextAreaElement;
    textarea.focus();

    fireEvent.click(screen.getByRole("button", { name: /^link$/i }));

    await waitFor(() => {
      expect(textarea.value).toBe("[link text](https://)");
    });
  });

  test("Preview toggle hides the textarea and renders the markdown bio", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ description: "**hi**" }),
      ),
    );

    render(<SettingsProfilePage />);

    expect(
      await screen.findByPlaceholderText(/tell people what you build/i),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /^preview$/i }));

    expect(
      screen.queryByPlaceholderText(/tell people what you build/i),
    ).toBeNull();
    expect(screen.getByText("hi").tagName.toLowerCase()).toBe("strong");
  });

  test("Preview shows fallback copy when bio is empty", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile({ description: "" })),
    );

    render(<SettingsProfilePage />);

    await screen.findByPlaceholderText(/tell people what you build/i);

    fireEvent.click(screen.getByRole("button", { name: /^preview$/i }));

    expect(screen.getByText(/nothing to preview yet/i)).toBeDefined();
  });

  test("Edit toggle returns control to the textarea", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ description: "see me" }),
      ),
    );

    render(<SettingsProfilePage />);

    await screen.findByPlaceholderText(/tell people what you build/i);

    fireEvent.click(screen.getByRole("button", { name: /^preview$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^edit$/i }));

    expect(
      (
        screen.getByPlaceholderText(
          /tell people what you build/i,
        ) as HTMLTextAreaElement
      ).value,
    ).toBe("see me");
  });
});

describe("SettingsProfilePage - links section", () => {
  test("Add link adds an empty slot up to the limit", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ links: ["https://a.dev"] }),
      ),
    );

    render(<SettingsProfilePage />);

    await screen.findByLabelText(/^link 1$/i);
    expect(screen.getByLabelText(/^link 2$/i)).toBeDefined();
    expect(screen.getByLabelText(/^link 3$/i)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /add link/i }));

    expect(await screen.findByLabelText(/^link 4$/i)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /add link/i }));

    expect(await screen.findByLabelText(/^link 5$/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /limit of 5 reached/i }),
    ).toBeDefined();
  });

  test("Remove link discards the slot at the given index", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({ links: ["https://a.dev", "https://b.dev"] }),
      ),
    );

    render(<SettingsProfilePage />);

    await screen.findByLabelText(/^link 1$/i);

    fireEvent.click(screen.getByRole("button", { name: /^remove link 1$/i }));

    await waitFor(() => {
      expect(
        (screen.getByLabelText(/^link 1$/i) as HTMLInputElement).value,
      ).toBe("https://b.dev");
    });
  });

  test("editing a link updates its value", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const link = (await screen.findByLabelText(
      /^link 1$/i,
    )) as HTMLInputElement;
    fireEvent.change(link, { target: { value: "https://changed.dev" } });

    expect(link.value).toBe("https://changed.dev");
  });
});

describe("SettingsProfilePage - save & discard", () => {
  test("Save and Discard are disabled when the form is pristine", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    await screen.findByLabelText(/display name/i);

    expect(
      (
        screen.getByRole("button", {
          name: /save changes/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: /discard/i }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  test("Discard reverts edits to the initial loaded values", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "New Name" } });

    expect(nameInput.value).toBe("New Name");

    fireEvent.click(screen.getByRole("button", { name: /discard/i }));

    await waitFor(() => {
      expect(nameInput.value).toBe("Jane Doe");
    });
  });

  test("Save submits the profile and shows a success toast", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile()),
      getPostV2UpdateUserProfileMockHandler200(),
    );

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "Jane Z" } });

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Profile saved",
          variant: "success",
        }),
      );
    });
  });

  test("Save shows a destructive toast on a 422", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile()),
      getPostV2UpdateUserProfileMockHandler422(),
    );

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "Jane Z" } });

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to save profile",
          variant: "destructive",
        }),
      );
    });
  });
});

describe("SettingsProfilePage - avatar upload", () => {
  test("uploading an avatar updates the form state", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile()),
      getPostV2UploadSubmissionMediaMockHandler200(
        "https://cdn.example.com/uploaded.png",
      ),
    );

    render(<SettingsProfilePage />);

    await screen.findByLabelText(/display name/i);

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    expect(fileInput).toBeTruthy();

    const file = new File(["x"], "avatar.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(
        (
          screen.getByRole("button", {
            name: /save changes/i,
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false);
    });
  });

  test("upload error surfaces a destructive toast", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(makeProfile()),
      getPostV2UploadSubmissionMediaMockHandler401(),
    );

    render(<SettingsProfilePage />);

    await screen.findByLabelText(/display name/i);

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const file = new File(["x"], "avatar.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to upload photo",
          variant: "destructive",
        }),
      );
    });
  });
});

describe("SettingsProfilePage - skeleton & nullish profile fields", () => {
  test("renders the skeleton on first render before the user resolves", () => {
    mockUseSupabase.mockReturnValue({
      user: null,
      isLoggedIn: false,
      isUserLoading: true,
      supabase: {},
    });

    const { container } = render(<SettingsProfilePage />);

    expect(
      screen.queryByRole("heading", { name: /^profile$/i, level: 1 }),
    ).toBeNull();
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(
      0,
    );
  });

  test("hydrates safely when the API returns nullish profile fields", async () => {
    server.use(
      getGetV2GetUserProfileMockHandler200(
        makeProfile({
          name: null as unknown as string,
          username: null as unknown as string,
          description: null as unknown as string,
          avatar_url: null,
          links: null as unknown as string[],
        }),
      ),
    );

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    expect(nameInput.value).toBe("");
    expect((screen.getByLabelText(/^handle$/i) as HTMLInputElement).value).toBe(
      "",
    );

    // 3 padded link slots
    expect(screen.getByLabelText(/^link 1$/i)).toBeDefined();
    expect(screen.getByLabelText(/^link 2$/i)).toBeDefined();
    expect(screen.getByLabelText(/^link 3$/i)).toBeDefined();
  });
});

describe("SettingsProfilePage - skeleton wrapper", () => {
  test("renders the skeleton component while the profile is loading", () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    const { container } = render(<SettingsProfilePage />);

    // Initial render — query is in-flight, skeleton shows
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(
      0,
    );

    // Heading appears once data resolves; verifies the skeleton path was taken first
    return screen.findByRole("heading", { name: /^profile$/i, level: 1 });
  });
});

describe("SettingsProfilePage - guard rails", () => {
  test("clicking Save while invalid does not fire the update endpoint", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const nameInput = (await screen.findByLabelText(
      /display name/i,
    )) as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "" } });

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    // Disabled button + invalid form: no toast, no network call
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Profile saved" }),
    );
  });

  test("renders the avatar change-photo button with proper a11y label", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    expect(
      await screen.findByRole("button", { name: /change profile photo/i }),
    ).toBeDefined();
  });

  test("links section header explains the cap", async () => {
    server.use(getGetV2GetUserProfileMockHandler200(makeProfile()));

    render(<SettingsProfilePage />);

    const links = await screen.findByText(/your links/i);
    const container = links.parentElement?.parentElement;
    expect(container).toBeTruthy();
    if (container) {
      expect(within(container).getByText(/add up to 5 links/i)).toBeDefined();
    }
  });
});
