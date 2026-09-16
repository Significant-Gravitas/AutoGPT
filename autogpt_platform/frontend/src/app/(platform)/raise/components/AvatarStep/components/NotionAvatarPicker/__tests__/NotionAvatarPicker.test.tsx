import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { NotionAvatarPicker } from "../NotionAvatarPicker";

describe("NotionAvatarPicker", () => {
  it("saves the color selected while an upload is still in flight, not the one queued before it", async () => {
    let resolveUpload: (() => void) | undefined;
    const pending = new Promise<void>((resolve) => {
      resolveUpload = resolve;
    });
    server.use(
      http.post("*/api/store/submissions/media", async () => {
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
    expect(onPick).toHaveBeenCalledWith(
      "https://cdn.test/uploaded.png",
      "blue-300",
    );
  });
});
