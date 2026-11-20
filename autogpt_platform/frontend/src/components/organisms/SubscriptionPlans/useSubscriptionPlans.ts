import { useState } from "react";
import type { PlanDialog } from "./helpers";

export function useSubscriptionPlans() {
  const [dialog, setDialog] = useState<PlanDialog>(null);

  function setDialogOpen(open: boolean) {
    if (!open) setDialog(null);
  }

  return { dialog, setDialog, setDialogOpen };
}
