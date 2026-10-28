"use client";

import { useState } from "react";

import { APIKeyList } from "./components/APIKeyList/APIKeyList";
import { APIKeysHeader } from "./components/APIKeysHeader/APIKeysHeader";
import { CreateAPIKeyDialog } from "./components/CreateAPIKeyDialog/CreateAPIKeyDialog";

export default function SettingsApiKeysPage() {
  const [createOpen, setCreateOpen] = useState(false);

  function openCreate() {
    setCreateOpen(true);
  }

  return (
    <>
      <APIKeysHeader onCreate={openCreate} />
      <APIKeyList />
      <CreateAPIKeyDialog open={createOpen} onOpenChange={setCreateOpen} />
    </>
  );
}
