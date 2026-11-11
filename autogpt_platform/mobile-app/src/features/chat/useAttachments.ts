import * as Crypto from "expo-crypto";
import { useState } from "react";

import { ApiError } from "@/lib/api/client";
import { uploadSessionFile } from "@/lib/api/workspace";

import {
  admitPicks,
  MAX_ATTACHMENTS,
  readyFiles,
  TOO_MANY_MESSAGE,
  uploadErrorMessage,
  type Attachment,
  type PickedFile,
} from "./attachmentHelpers";
import { chooseAttachSource, pickFrom } from "./pickFiles";

type Item = Attachment & PickedFile;

/** Files picked for the next message; each uploads at once, into the chat's folder when there is a chat. */
export function useAttachments(sessionID: string | null) {
  const [items, setItems] = useState<Item[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  function update(id: string, patch: Partial<Item>) {
    // A removed or sent item is gone from the list, so a late result is dropped.
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  }

  async function upload(item: Item) {
    update(item.id, { status: "uploading", error: undefined });
    try {
      const uploaded = await uploadSessionFile(sessionID, item);
      update(item.id, { status: "ready", fileID: uploaded.file_id });
    } catch (error) {
      const message =
        error instanceof ApiError
          ? uploadErrorMessage(error.status, error.message)
          : uploadErrorMessage(null);
      update(item.id, { status: "failed", error: message });
    }
  }

  async function attach() {
    const source = await chooseAttachSource();
    if (!source) return;
    const room = MAX_ATTACHMENTS - items.length;
    if (room <= 0) {
      setNotice(TOO_MANY_MESSAGE);
      return;
    }
    const picked = await pickFrom(source, room).catch(() => []);
    const { accepted, notice: limitNotice } = admitPicks(items.length, picked);
    setNotice(limitNotice);
    const fresh: Item[] = accepted.map((file) => ({
      ...file,
      id: Crypto.randomUUID(),
      status: "uploading",
    }));
    setItems((prev) => [...prev, ...fresh]);
    for (const item of fresh) void upload(item);
  }

  function retry(id: string) {
    const item = items.find((entry) => entry.id === id);
    if (item?.status === "failed") void upload(item);
  }

  function remove(id: string) {
    setNotice(null);
    setItems((prev) => prev.filter((item) => item.id !== id));
  }

  function clear() {
    setNotice(null);
    setItems([]);
  }

  const failure = items.find((item) => item.status === "failed")?.error;

  return {
    attachments: items as Attachment[],
    readyFiles: readyFiles(items),
    message: notice ?? failure ?? null,
    attach,
    retry,
    remove,
    clear,
  };
}
