import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { saveBlob } from "../save-blob";
import { nativeDownloadMimeType } from "../native-download-protocol";

describe("browser blob download", () => {
  beforeEach(() => {
    delete window.AutoGPTDownloads;
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test-download");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retains the filename and cleans the attached anchor and object URL", async () => {
    let filename = "";
    let connected = false;
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      filename = this.download;
      connected = this.isConnected;
    });
    const blob = new Blob(["Hello"], { type: "text/plain" });
    await saveBlob(blob, "report.txt");
    expect(filename).toBe("report.txt");
    expect(connected).toBe(true);
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-download");
    expect(document.querySelector('a[href="blob:test-download"]')).toBeNull();
  });

  it("cleans up when the browser download throws", async () => {
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {
      throw new Error("Download failed");
    });
    await expect(saveBlob(new Blob(["x"]), "x.txt")).rejects.toThrow(
      "Download failed",
    );
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-download");
    expect(document.querySelector('a[href="blob:test-download"]')).toBeNull();
  });

  it("does not start an already cancelled download", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      saveBlob(new Blob(["x"]), "x.txt", { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
});

it("replaces overlong MIME metadata with the native download fallback", () => {
  expect(nativeDownloadMimeType(`application/${"a".repeat(130)}`)).toBe(
    "application/octet-stream",
  );
});
