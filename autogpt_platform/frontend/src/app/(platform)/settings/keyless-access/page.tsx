"use client";

import { useState } from "react";

import { CreateWorkloadDialog } from "./components/CreateWorkloadDialog/CreateWorkloadDialog";
import { HowItWorks } from "./components/HowItWorks/HowItWorks";
import { KeylessAccessHeader } from "./components/KeylessAccessHeader/KeylessAccessHeader";
import { WorkloadList } from "./components/WorkloadList/WorkloadList";

export default function SettingsKeylessAccessPage() {
  const [createOpen, setCreateOpen] = useState(false);

  function openCreate() {
    setCreateOpen(true);
  }

  return (
    <>
      <KeylessAccessHeader onCreate={openCreate} />
      <div className="flex flex-col gap-6">
        <HowItWorks />
        <WorkloadList onCreate={openCreate} />
      </div>
      <CreateWorkloadDialog open={createOpen} onOpenChange={setCreateOpen} />
    </>
  );
}
