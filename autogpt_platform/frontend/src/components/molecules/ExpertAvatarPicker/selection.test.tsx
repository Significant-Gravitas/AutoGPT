import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";
import { defaultAvatarUrl } from "./helpers";

function pendingGeneration() {
  return [
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json({ id: "pending", status: "pending" }, { status: 202 }),
    ),
    http.get("*/api/experts/avatars/generations/pending", () =>
      HttpResponse.json({ id: "pending", status: "pending" }),
    ),
  ];
}

test("can keep the default while generation is pending", async () => {
  const onPick = vi.fn();
  server.use(...pendingGeneration());
  render(<ExpertAvatarPicker name="Nova" category="finance" onPick={onPick} />);
  await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
  await screen.findByRole("status");
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(defaultAvatarUrl("finance", "Nova"));
  expect(screen.queryByRole("status")).toBeNull();
});

test("keeps an uploaded preview through a failed regeneration", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/store/submissions/media", () =>
      HttpResponse.json("https://cdn.test/upload.png"),
    ),
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json({ detail: "Try again later" }, { status: 429 }),
    ),
  );
  render(<ExpertAvatarPicker name="Nova" category="finance" onPick={onPick} />);
  await userEvent.upload(
    screen.getByLabelText("Upload avatar"),
    new File(["png"], "avatar.png", { type: "image/png" }),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
    ).toContain("upload.png"),
  );
  await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
  await screen.findByRole("alert");
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith("https://cdn.test/upload.png");
});

test("a late generation response cannot replace an upload", async () => {
  const onPick = vi.fn();
  const poll = vi.fn();
  let release = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let respond = () => {};
  const responded = new Promise<void>((resolve) => {
    respond = resolve;
  });
  server.use(
    http.post("*/api/experts/avatars/generations", async () => {
      await pending;
      return HttpResponse.json(
        { id: "late", status: "pending" },
        { status: 202 },
      );
    }),
    http.get("*/api/experts/avatars/generations/late", () => {
      poll();
      return HttpResponse.json({
        id: "late",
        status: "complete",
        avatar_url: "https://cdn.test/late.png",
      });
    }),
    http.post("*/api/store/submissions/media", () =>
      HttpResponse.json("https://cdn.test/upload.png"),
    ),
  );
  function onResponse({ request }: { request: Request }) {
    if (request.url.endsWith("/avatars/generations")) respond();
  }
  server.events.on("response:mocked", onResponse);
  try {
    render(
      <ExpertAvatarPicker name="Nova" category="finance" onPick={onPick} />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
    await screen.findByRole("status");
    await userEvent.upload(
      screen.getByLabelText("Upload avatar"),
      new File(["png"], "avatar.png", { type: "image/png" }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
      ).toContain("upload.png"),
    );
    await act(async () => {
      release();
      await responded;
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Use this avatar" }),
    );
    expect(onPick).toHaveBeenCalledWith("https://cdn.test/upload.png");
    expect(screen.queryByRole("status")).toBeNull();
    expect(poll).not.toHaveBeenCalled();
  } finally {
    release();
    server.events.removeListener("response:mocked", onResponse);
  }
});

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));
