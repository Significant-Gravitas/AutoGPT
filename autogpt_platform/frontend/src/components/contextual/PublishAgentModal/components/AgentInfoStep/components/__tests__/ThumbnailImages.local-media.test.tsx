import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { ThumbnailImages } from "../ThumbnailImages";

vi.mock("next/image", async () => {
  return vi.importActual<typeof import("next/image")>("next/image");
});

const uploadSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/direct-upload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/direct-upload")>();
  return { ...actual, uploadSubmissionMediaDirect: uploadSpy };
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("local marketplace image previews", () => {
  it.each([
    "http://localhost:8006/api/store/media/user-1/images/thumbnail.png",
    "https://appliance.example/_agpt/api/store/media/user-1/images/thumbnail.png",
  ])("shows the uploaded image directly from %s", async (url) => {
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", url.split("/store/media/")[0]);
    uploadSpy.mockResolvedValueOnce(url);
    const onImagesChange = vi.fn();
    render(
      <ThumbnailImages agentId="agent-1" onImagesChange={onImagesChange} />,
    );

    const input = document.getElementById("image-upload") as HTMLInputElement;
    fireEvent.change(input, {
      target: {
        files: [new File(["image"], "thumbnail.png", { type: "image/png" })],
      },
    });

    const image = await screen.findByRole("img", { name: "Thumbnail 1" });
    expect(image.getAttribute("src")).toBe(url);
    expect(image.hasAttribute("srcset")).toBe(false);
    expect(onImagesChange).toHaveBeenCalledWith([url]);
  });

  it("keeps GCS thumbnails optimized", () => {
    const url = "https://storage.googleapis.com/media/thumbnail.png";
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={() => {}}
        initialImages={[url]}
      />,
    );

    const image = screen.getByRole("img", { name: "Thumbnail 1" });
    expect(image.getAttribute("src")).toMatch(/^\/_next\/image\?/);
    expect(image.getAttribute("srcset")).toContain("/_next/image?");
  });

  it("does not bypass the optimizer for an external host using the media path", () => {
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "http://localhost:8006/api");
    render(
      <ThumbnailImages
        agentId="agent-1"
        onImagesChange={() => {}}
        initialImages={[
          "https://untrusted.example/api/store/media/user-1/images/thumbnail.png",
        ]}
      />,
    );

    const image = screen.getByRole("img", { name: "Thumbnail 1" });
    expect(image.getAttribute("src")).toMatch(/^\/_next\/image\?/);
    expect(image.getAttribute("srcset")).toContain("/_next/image?");
  });
});
