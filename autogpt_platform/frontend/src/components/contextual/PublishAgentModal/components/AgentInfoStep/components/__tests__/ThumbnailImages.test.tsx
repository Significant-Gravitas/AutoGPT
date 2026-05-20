import { describe, expect, it, vi } from "vitest";

import {
  getPostV2GenerateSubmissionImageMockHandler,
  getPostV2UploadSubmissionMediaMockHandler,
  getPostV2UploadSubmissionMediaMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import { ThumbnailImages } from "../ThumbnailImages";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({
      toast: toastSpy,
      dismiss: () => {},
      toasts: [],
    }),
  };
});

describe("ThumbnailImages", () => {
  it("renders the empty state with an Add image label and Generate button", () => {
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={() => {}}
        initialImages={[]}
      />,
    );

    expect(screen.getByTestId("thumbnail-add-image-empty")).toBeDefined();
    expect(screen.getByRole("button", { name: /generate/i })).toBeDefined();
  });

  it("renders existing thumbnails with a remove button for each, including the first", () => {
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={() => {}}
        initialImages={["https://cdn.test/a.png", "https://cdn.test/b.png"]}
        initialSelectedImage="https://cdn.test/a.png"
      />,
    );

    expect(screen.getByTestId("thumbnail-remove-0")).toBeDefined();
    expect(screen.getByTestId("thumbnail-remove-1")).toBeDefined();
  });

  it("removes the first image when its remove button is clicked", () => {
    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={onImagesChange}
        initialImages={["https://cdn.test/a.png", "https://cdn.test/b.png"]}
        initialSelectedImage="https://cdn.test/a.png"
      />,
    );

    fireEvent.click(screen.getByTestId("thumbnail-remove-0"));

    expect(screen.queryByTestId("thumbnail-remove-1")).toBeNull();
    expect(onImagesChange).toHaveBeenCalledWith(["https://cdn.test/b.png"]);
  });

  it("uploads a file via the empty-state input and shows the new image", async () => {
    server.use(
      getPostV2UploadSubmissionMediaMockHandler(
        "https://cdn.test/uploaded.png",
      ),
    );

    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={onImagesChange}
        initialImages={[]}
      />,
    );

    const input = document.getElementById("image-upload") as HTMLInputElement;
    const file = new File(["data"], "thumb.png", { type: "image/png" });
    Object.defineProperty(input, "files", { value: [file] });
    fireEvent.change(input);

    await waitFor(() => {
      expect(onImagesChange).toHaveBeenCalledWith([
        "https://cdn.test/uploaded.png",
      ]);
    });
  });

  it("shows a destructive toast when the upload endpoint fails", async () => {
    toastSpy.mockClear();
    server.use(getPostV2UploadSubmissionMediaMockHandler422());

    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={() => {}}
        initialImages={[]}
      />,
    );

    const input = document.getElementById("image-upload") as HTMLInputElement;
    const file = new File(["data"], "thumb.png", { type: "image/png" });
    Object.defineProperty(input, "files", { value: [file] });
    fireEvent.change(input);

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Upload failed",
          variant: "destructive",
        }),
      );
    });
  });

  it("generates an image when the Generate button is clicked", async () => {
    server.use(
      getPostV2GenerateSubmissionImageMockHandler({
        image_url: "https://cdn.test/generated.png",
      }),
    );

    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={onImagesChange}
        initialImages={[]}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    await waitFor(() => {
      expect(onImagesChange).toHaveBeenCalledWith([
        "https://cdn.test/generated.png",
      ]);
    });
  });

  it("clears the visible thumbnails when the only image is removed", () => {
    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={onImagesChange}
        initialImages={["https://cdn.test/only.png"]}
        initialSelectedImage="https://cdn.test/only.png"
      />,
    );

    fireEvent.click(screen.getByTestId("thumbnail-remove-0"));

    expect(screen.queryByTestId("thumbnail-remove-0")).toBeNull();
    expect(onImagesChange).toHaveBeenLastCalledWith([]);
    expect(screen.getByTestId("thumbnail-add-image-empty")).toBeDefined();
  });

  it("toasts a generation error when no agentId is provided", async () => {
    toastSpy.mockClear();
    render(
      <ThumbnailImages
        agentId={null}
        onImagesChange={() => {}}
        initialImages={[]}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Generation failed",
          variant: "destructive",
        }),
      );
    });
  });

  it("makes the clicked thumbnail the new selected (first) image", () => {
    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={onImagesChange}
        initialImages={["https://cdn.test/a.png", "https://cdn.test/b.png"]}
        initialSelectedImage="https://cdn.test/a.png"
      />,
    );

    const secondButton = screen.getByRole("button", {
      name: /select thumbnail 2/i,
    });
    fireEvent.click(secondButton);

    expect(onImagesChange).toHaveBeenLastCalledWith([
      "https://cdn.test/b.png",
      "https://cdn.test/a.png",
    ]);
  });
});
