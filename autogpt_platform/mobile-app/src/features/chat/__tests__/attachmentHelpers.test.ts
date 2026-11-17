import {
  admitPicks,
  MAX_ATTACHMENT_BYTES,
  messageFiles,
  nameFromUri,
  readyFiles,
  REJECTED_MESSAGE,
  STORAGE_FULL_MESSAGE,
  TOO_LARGE_MESSAGE,
  TOO_MANY_MESSAGE,
  UPLOAD_FAILED_MESSAGE,
  uploadErrorMessage,
  type Attachment,
} from "../attachmentHelpers";
import { PLAN_REQUIRED_MESSAGE } from "../chatErrors";

function picks(count: number, size = 1000) {
  return Array.from({ length: count }, (_, index) => ({ name: `f${index}`, size }));
}

describe("admitPicks", () => {
  it("takes every pick that fits", () => {
    const result = admitPicks(2, picks(3));
    expect(result.accepted).toHaveLength(3);
    expect(result.notice).toBeNull();
  });

  it("stops at ten attachments in all", () => {
    const result = admitPicks(8, picks(4));
    expect(result.accepted.map((pick) => pick.name)).toEqual(["f0", "f1"]);
    expect(result.notice).toBe(TOO_MANY_MESSAGE);
  });

  it("leaves out a file over 25 MB and keeps the rest", () => {
    const big = { name: "big", size: MAX_ATTACHMENT_BYTES + 1 };
    const result = admitPicks(0, [big, ...picks(1)]);
    expect(result.accepted.map((pick) => pick.name)).toEqual(["f0"]);
    expect(result.notice).toBe(TOO_LARGE_MESSAGE);
  });

  it("lets a file of unknown size through", () => {
    const unknown: { name: string; size?: number } = { name: "x" };
    expect(admitPicks(0, [unknown]).accepted).toHaveLength(1);
  });
});

describe("uploadErrorMessage", () => {
  it("tells a full quota from a file that is too large", () => {
    expect(uploadErrorMessage(413, "Storage limit exceeded: 1 GB")).toBe(
      STORAGE_FULL_MESSAGE,
    );
    expect(uploadErrorMessage(413, "File exceeds maximum size of 100 MB")).toBe(
      TOO_LARGE_MESSAGE,
    );
  });

  it("names a scan rejection, a missing plan, and anything else", () => {
    expect(uploadErrorMessage(400, "Virus detected: EICAR")).toBe(REJECTED_MESSAGE);
    expect(uploadErrorMessage(402)).toBe(PLAN_REQUIRED_MESSAGE);
    expect(uploadErrorMessage(500)).toBe(UPLOAD_FAILED_MESSAGE);
    expect(uploadErrorMessage(409)).toBe(UPLOAD_FAILED_MESSAGE);
    expect(uploadErrorMessage(null)).toBe(UPLOAD_FAILED_MESSAGE);
  });
});

describe("nameFromUri", () => {
  it("takes the decoded last segment", () => {
    expect(nameFromUri("file:///tmp/My%20Photo.jpg", "photo")).toBe("My Photo.jpg");
  });

  it("falls back when there is none", () => {
    expect(nameFromUri("file:///tmp/", "photo")).toBe("photo");
  });
});

describe("readyFiles", () => {
  it("keeps only uploaded files", () => {
    const items: Attachment[] = [
      { id: "1", name: "a.pdf", isImage: false, status: "ready", fileID: "f1" },
      { id: "2", name: "b.pdf", isImage: false, status: "uploading" },
      { id: "3", name: "c.pdf", isImage: false, status: "failed" },
    ];
    expect(readyFiles(items)).toEqual([{ id: "f1", name: "a.pdf" }]);
  });
});

describe("messageFiles", () => {
  it("reads ids and names from the metadata", () => {
    expect(messageFiles({ fileIDs: ["f1"], fileNames: ["a.pdf"] })).toEqual({
      fileIDs: ["f1"],
      fileNames: ["a.pdf"],
    });
  });

  it("is empty for anything else", () => {
    expect(messageFiles(undefined)).toEqual({ fileIDs: [], fileNames: [] });
    expect(messageFiles({ fileIDs: "f1" })).toEqual({ fileIDs: [], fileNames: [] });
  });
});
