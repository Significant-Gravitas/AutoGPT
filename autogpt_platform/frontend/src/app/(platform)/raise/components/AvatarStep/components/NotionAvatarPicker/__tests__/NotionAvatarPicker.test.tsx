import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { NotionAvatarPicker } from "../NotionAvatarPicker";

describe("NotionAvatarPicker", () => {
  it("saves the color selected while an upload is still in flight, not the one queued before it", async () => {
    let purpose: string | null = null;
    let resolveUpload: (() => void) | undefined;
    const pending = new Promise<void>((resolve) => {
      resolveUpload = resolve;
    });
    server.use(
      http.post("*/api/store/submissions/media", async ({ request }) => {
        purpose = new URL(request.url).searchParams.get("purpose");
        await pending;
        return HttpResponse.json("https://cdn.test/uploaded.png");
      }),
    );
    const onPick = vi.fn();
    const user = userEvent.setup();

    render(
      <NotionAvatarPicker name="Maria" color="rose-300" onPick={onPick} />,
    );

    const file = new File(["x"], "avatar.png", { type: "image/png" });
    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    await user.upload(fileInput, file);

    // Change the color while the upload is still pending — the color
    // controls aren't disabled during upload, so this is a real user path.
    await user.click(screen.getByRole("button", { name: "Blue" }));

    resolveUpload?.();

    await waitFor(() => expect(onPick).toHaveBeenCalledTimes(1));
    expect(purpose).toBe("expert-avatar");
    expect(onPick).toHaveBeenCalledWith(
      "https://cdn.test/uploaded.png",
      "blue-300",
    );
  });
});

it.each([400, 422, 503])(
  "preserves the draft and explains an upload rejection (%s)",
  async (status) => {
    server.use(
      http.post("*/api/store/submissions/media", () =>
        HttpResponse.json(
          { detail: "Review unavailable. Please try again later." },
          { status },
        ),
      ),
    );
    const onPick = vi.fn();
    render(
      <>
        <NotionAvatarPicker name="Maria" color="rose-300" onPick={onPick} />
        <Toaster />
      </>,
    );
    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(["x"], "avatar.png", { type: "image/png" }),
    );
    expect(
      await screen.findByText("Review unavailable. Please try again later."),
    ).toBeDefined();
    expect(onPick).not.toHaveBeenCalled();
  },
);

it("does not offer or upload GIF appearances", async () => {
  const upload = vi.fn(() => HttpResponse.json("https://cdn.test/avatar.gif"));
  server.use(http.post("*/api/store/submissions/media", upload));
  render(
    <>
      <NotionAvatarPicker name="Maria" color="rose-300" onPick={vi.fn()} />
      <Toaster />
    </>,
  );
  const input = document.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  expect(input.accept).not.toContain("image/gif");
  await userEvent
    .setup({ applyAccept: false })
    .upload(input, new File(["GIF89a"], "avatar.gif", { type: "image/gif" }));
  expect(
    await screen.findByText("Pick a PNG, JPEG, or WebP image."),
  ).toBeDefined();
  expect(upload).not.toHaveBeenCalled();
});
