import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

const FINANCE_ARTWORK = "/experts/clay/v1/finance.png";

function completesAs(url: string, requests: unknown[] = []) {
  return [
    http.post("*/api/experts/avatars/generations", async ({ request }) => {
      requests.push(await request.json());
      return HttpResponse.json(
        { id: "job-1", status: "pending" },
        { status: 202 },
      );
    }),
    http.get("*/api/experts/avatars/generations/job-1", () =>
      HttpResponse.json({
        id: "job-1",
        status: "complete",
        avatar_url: url,
      }),
    ),
  ];
}

test("auto-generates on mount, showing the category artwork until it lands", async () => {
  const onPick = vi.fn();
  server.use(...completesAs("https://cdn.test/generated.png"));
  render(
    <ExpertAvatarPicker
      name="Nova"
      category="finance"
      autoGenerate
      onPick={onPick}
    />,
  );
  await screen.findByRole("status");
  expect(
    screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
  ).toContain("finance.png");

  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
    ).toContain("generated.png"),
  );
  expect(onPick).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith("https://cdn.test/generated.png");
});

test("regenerating rolls every trait but the category", async () => {
  const requests: unknown[] = [];
  server.use(...completesAs("https://cdn.test/generated.png", requests));
  render(
    <ExpertAvatarPicker name="Nova" category="marketing" onPick={vi.fn()} />,
  );
  await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
  await waitFor(() => expect(requests).toHaveLength(1));
  await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
  await waitFor(() => expect(requests).toHaveLength(2));

  for (const request of requests) {
    const body = request as Record<string, string>;
    expect(body.category).toBe("marketing");
    expect(Object.keys(body).sort()).toEqual([
      "accent_count",
      "accent_placement",
      "base",
      "category",
      "expression",
      "inlay",
      "shade",
      "shape",
      "tilt",
    ]);
  }
});

test("a failed generation leaves the category artwork ready to keep", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json({ id: "job-2", status: "pending" }, { status: 202 }),
    ),
    http.get("*/api/experts/avatars/generations/job-2", () =>
      HttpResponse.json({
        id: "job-2",
        status: "failed",
        error: "Could not generate avatar",
      }),
    ),
  );
  render(
    <ExpertAvatarPicker
      name="Nova"
      category="finance"
      autoGenerate
      onPick={onPick}
    />,
  );
  expect((await screen.findByRole("alert")).textContent).toContain(
    "Could not generate avatar",
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(FINANCE_ARTWORK);
});

test("an existing avatar is kept until a generation replaces it", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/experts/avatars/generations", () =>
      HttpResponse.json(
        { detail: "Please wait before generating again." },
        { status: 429 },
      ),
    ),
  );
  render(
    <ExpertAvatarPicker
      name="Nova"
      category="finance"
      avatarUrl="https://cdn.test/existing.png"
      onPick={onPick}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
  await screen.findByRole("alert");
  expect(
    screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
  ).toContain("existing.png");
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith("https://cdn.test/existing.png");
});

test("uploads remain previews until chosen", async () => {
  const onPick = vi.fn();
  server.use(
    http.post("*/api/store/submissions/media", () =>
      HttpResponse.json("https://cdn.test/upload.png"),
    ),
  );
  render(<ExpertAvatarPicker name="Nova" category="support" onPick={onPick} />);
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
  expect(onPick).toHaveBeenCalledWith("https://cdn.test/upload.png");
});

test("rejects oversized uploads before making a request", async () => {
  const upload = vi.fn();
  server.use(
    http.post("*/api/store/submissions/media", () => {
      upload();
      return HttpResponse.json("https://cdn.test/upload.png");
    }),
  );
  render(
    <ExpertAvatarPicker name="Nova" category="content" onPick={vi.fn()} />,
  );
  const file = new File(["png"], "avatar.png", { type: "image/png" });
  Object.defineProperty(file, "size", { value: 6 * 1024 * 1024 });
  await userEvent.upload(screen.getByLabelText("Upload avatar"), file);
  expect((await screen.findByRole("alert")).textContent).toContain("under 5MB");
  expect(upload).not.toHaveBeenCalled();
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
  render(
    <ExpertAvatarPicker
      name="Nova"
      category="content"
      autoGenerate
      onPick={vi.fn()}
    />,
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

test("offers one avatar with no catalog or trait controls", () => {
  render(
    <ExpertAvatarPicker name="Nova" category="content" onPick={vi.fn()} />,
  );
  expect(screen.queryByRole("group", { name: "Avatar catalog" })).toBeNull();
  expect(screen.queryByRole("combobox")).toBeNull();
  expect(screen.queryByRole("button", { name: "Sage" })).toBeNull();
});

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));
