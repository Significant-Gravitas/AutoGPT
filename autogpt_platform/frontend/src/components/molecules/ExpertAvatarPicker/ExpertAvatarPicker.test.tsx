import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

test("generates a preview and only saves the selected PNG after confirmation", async () => {
  const onPick = vi.fn();
  const requests: unknown[] = [];
  server.use(
    http.post("*/api/experts/avatars/generations", async ({ request }) => {
      requests.push(await request.json());
      return HttpResponse.json(
        { id: "job-1", status: "pending", created_at: Date.now() / 1000 },
        { status: 202 },
      );
    }),
    http.get("*/api/experts/avatars/generations/job-1", () =>
      HttpResponse.json({
        id: "job-1",
        status: "complete",
        avatar_url: "https://cdn.test/generated.png",
        created_at: Date.now() / 1000,
      }),
    ),
  );
  render(<ExpertAvatarPicker name="Nova" color={null} onPick={onPick} />);
  await userEvent.click(screen.getByRole("button", { name: "Sage" }));
  await userEvent.click(
    screen.getByRole("button", { name: "Generate with AI" }),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
    ).toContain("generated.png"),
  );
  expect(requests).toEqual([
    {
      category: "finance",
      shade: "standard",
      shape: "pebble",
      base: "compact",
      tilt: "level",
      inlay: "sweep",
      accent_placement: "body",
      accent_count: "one",
      expression: "friendly",
    },
  ]);
  expect(onPick).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(
    "https://cdn.test/generated.png",
    "green-300",
  );
});

test("a failed job keeps the current avatar and the catalog remains usable", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json(
        { id: "job-2", status: "pending", created_at: Date.now() / 1000 },
        { status: 202 },
      ),
    ),
    http.get("*/api/experts/avatars/generations/job-2", () =>
      HttpResponse.json({
        id: "job-2",
        status: "failed",
        error: "Could not generate avatar",
        created_at: Date.now() / 1000,
      }),
    ),
  );
  render(
    <ExpertAvatarPicker
      name="Nova"
      color="rose-300"
      avatarUrl="https://cdn.test/existing.png"
      onPick={onPick}
    />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Generate with AI" }),
  );
  expect((await screen.findByRole("alert")).textContent).toContain(
    "Could not generate avatar",
  );
  expect(
    screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
  ).toContain("existing.png");
  expect(onPick).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Ochre" }));
  await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(
    "/experts/clay/v1/sales.png",
    "amber-300",
  );
});

test("uploads remain previews until chosen", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/store/submissions/media", () =>
      HttpResponse.json("https://cdn.test/upload.png"),
    ),
  );
  render(<ExpertAvatarPicker name="Nova" color="rose-300" onPick={onPick} />);
  await userEvent.upload(
    screen.getByLabelText("Upload avatar"),
    new File(["png"], "avatar.png", { type: "image/png" }),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
    ).toContain("upload.png"),
  );
  expect(onPick).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(
    "https://cdn.test/upload.png",
    "rose-300",
  );
});

test("rejects oversized uploads before making a request", async () => {
  const upload = vi.fn();
  server.use(
    http.post("*/api/store/submissions/media", () => {
      upload();
      return HttpResponse.json("https://cdn.test/upload.png");
    }),
  );
  render(<ExpertAvatarPicker name="Nova" color={null} onPick={vi.fn()} />);
  const file = new File(["png"], "avatar.png", { type: "image/png" });
  Object.defineProperty(file, "size", { value: 6 * 1024 * 1024 });
  await userEvent.upload(screen.getByLabelText("Upload avatar"), file);
  expect((await screen.findByRole("alert")).textContent).toContain("under 5MB");
  expect(upload).not.toHaveBeenCalled();
});

test.each([
  "/experts/clay/v1/finance.png",
  "/autogpt-characters/v1.1/expert-mina/neutral/128.webp",
])("editing %s uses its color for generation", async (avatarUrl) => {
  const requests: unknown[] = [];
  server.use(
    http.post("*/api/experts/avatars/generations", async ({ request }) => {
      requests.push(await request.json());
      return HttpResponse.json(
        { detail: "Please wait before generating again." },
        { status: 429 },
      );
    }),
  );
  render(
    <ExpertAvatarPicker
      name="Nova"
      color="green-300"
      avatarUrl={avatarUrl}
      onPick={vi.fn()}
    />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Generate with AI" }),
  );
  await screen.findByRole("alert");
  expect(requests).toEqual([
    {
      category: "finance",
      shade: "standard",
      shape: "pebble",
      base: "compact",
      tilt: "level",
      inlay: "sweep",
      accent_placement: "body",
      accent_count: "one",
      expression: "friendly",
    },
  ]);
  expect(
    screen.getByRole("img", { name: /^Nova/ }).getAttribute("src"),
  ).toContain(
    avatarUrl.includes("expert-mina") ? "expert-mina" : "finance.png",
  );
});

test("a pending generation disables uploads and confirmation", async () => {
  server.use(
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json(
        { id: "pending-job", status: "pending" },
        { status: 202 },
      ),
    ),
    http.get("*/api/experts/avatars/generations/pending-job", () =>
      HttpResponse.json({ id: "pending-job", status: "pending" }),
    ),
  );
  render(<ExpertAvatarPicker name="Nova" color={null} onPick={vi.fn()} />);
  await userEvent.click(
    screen.getByRole("button", { name: "Generate with AI" }),
  );
  await screen.findByRole("status");
  expect(
    (screen.getByLabelText("Upload avatar") as HTMLInputElement).disabled,
  ).toBe(true);
  expect(
    (
      screen.getByRole("button", {
        name: "Use this avatar",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
});

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));
