import { afterEach, describe, expect, test, vi } from "vitest";
import { downloadFile, filenameFromContentDisposition } from "../download-file";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("downloadFile", () => {
  test("defers URL.revokeObjectURL so the queued download still resolves", () => {
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock");
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    vi.useFakeTimers();

    downloadFile("frankie.expert.zip", new Blob(["x"]));

    expect(createUrl).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
    // Revoking in the same tick can leave the browser fetching a dead URL.
    expect(revokeUrl).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(revokeUrl).toHaveBeenCalledWith("blob:mock");
  });

  test("names the file and leaves no anchor behind", () => {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    let downloadAttr = "";
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      downloadAttr = this.download;
    });

    downloadFile("frankie.expert.zip", new Blob(["x"]));

    expect(downloadAttr).toBe("frankie.expert.zip");
    expect(document.querySelectorAll("a").length).toBe(0);
  });
});

describe("filenameFromContentDisposition", () => {
  test("prefers RFC 5987 filename* over the ascii fallback beside it", () => {
    const headers = new Headers({
      "content-disposition":
        "attachment; filename=\"frankie.zip\"; filename*=UTF-8''fran%C3%A7ois.zip",
    });
    expect(filenameFromContentDisposition(headers, "fallback.zip")).toBe(
      "françois.zip",
    );
  });

  test("falls back to the plain filename when the escape is malformed", () => {
    const headers = new Headers({
      "content-disposition":
        "attachment; filename=\"frankie.zip\"; filename*=UTF-8''%E0%A4%A",
    });
    expect(filenameFromContentDisposition(headers, "fallback.zip")).toBe(
      "frankie.zip",
    );
  });

  test("accepts an unquoted filename", () => {
    const headers = new Headers({
      "content-disposition": "attachment; filename=frankie.zip",
    });
    expect(filenameFromContentDisposition(headers, "fallback.zip")).toBe(
      "frankie.zip",
    );
  });

  test("uses our own name when the server picked none", () => {
    expect(filenameFromContentDisposition(new Headers(), "fallback.zip")).toBe(
      "fallback.zip",
    );
  });
});
