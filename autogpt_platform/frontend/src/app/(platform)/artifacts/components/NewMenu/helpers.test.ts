import { beforeEach, describe, expect, test, vi } from "vitest";

const { uploadFileDirectMock, bulkMoveMock } = vi.hoisted(() => ({
  uploadFileDirectMock: vi.fn(),
  bulkMoveMock: vi.fn(),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: uploadFileDirectMock,
}));

vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  bulkMoveWorkspaceFiles: bulkMoveMock,
}));

import { uploadFiles } from "./helpers";

function makeFile(name: string) {
  return new File(["x"], name, { type: "text/plain" });
}

describe("uploadFiles", () => {
  beforeEach(() => {
    uploadFileDirectMock.mockReset();
    bulkMoveMock.mockReset();
    uploadFileDirectMock.mockImplementation(async (file: File) => ({
      file_id: `id-${file.name}`,
      name: file.name,
      path: `/${file.name}`,
      mime_type: file.type,
      size_bytes: file.size,
    }));
    bulkMoveMock.mockResolvedValue({ status: 200, data: [] });
  });

  test("uploads to the root without moving when no folder is selected", async () => {
    const outcome = await uploadFiles(
      [makeFile("a.txt"), makeFile("b.txt")],
      null,
    );

    expect(outcome).toEqual({ uploaded: 2, leftAtRoot: [], failed: [] });
    expect(bulkMoveMock).not.toHaveBeenCalled();
  });

  test("moves each upload into the selected folder", async () => {
    const outcome = await uploadFiles([makeFile("a.txt")], "fld-1");

    expect(outcome.uploaded).toBe(1);
    expect(bulkMoveMock).toHaveBeenCalledWith(
      {
        file_ids: ["id-a.txt"],
        folder_id: "fld-1",
      },
      undefined,
    );
  });

  test("pins upload and folder move to the same explicit tenant", async () => {
    const file = makeFile("team.txt");
    const scope = { organizationId: "org-files", teamId: "team-files" };
    await uploadFiles([file], "folder-team", scope);
    expect(uploadFileDirectMock).toHaveBeenCalledWith(file, undefined, scope);
    expect(bulkMoveMock).toHaveBeenCalledWith(
      { file_ids: ["id-team.txt"], folder_id: "folder-team" },
      { headers: { "X-Org-Id": "org-files", "X-Team-Id": "team-files" } },
    );
  });

  test("keeps a file counted as uploaded when the move fails", async () => {
    bulkMoveMock.mockResolvedValueOnce({ status: 422, data: {} });

    const outcome = await uploadFiles([makeFile("a.txt")], "fld-1");

    expect(outcome).toEqual({ uploaded: 1, leftAtRoot: ["a.txt"], failed: [] });
  });

  test("treats a thrown move as left at root", async () => {
    bulkMoveMock.mockRejectedValueOnce(new Error("network"));

    const outcome = await uploadFiles([makeFile("a.txt")], "fld-1");

    expect(outcome.leftAtRoot).toEqual(["a.txt"]);
    expect(outcome.failed).toEqual([]);
  });

  test("reports upload errors per file and keeps going", async () => {
    uploadFileDirectMock.mockRejectedValueOnce(new Error("Too large"));

    const outcome = await uploadFiles(
      [makeFile("big.bin"), makeFile("ok.txt")],
      null,
    );

    expect(outcome.uploaded).toBe(1);
    expect(outcome.failed).toEqual([{ name: "big.bin", message: "Too large" }]);
  });
});
