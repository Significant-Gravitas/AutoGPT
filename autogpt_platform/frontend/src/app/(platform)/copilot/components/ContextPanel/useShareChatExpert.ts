import { useEffect } from "react";
import { useCopilotUIStore, type ContextPanelExpert } from "../../store";

export function useShareChatExpert(expert: ContextPanelExpert | null) {
  const expertId = expert?.id ?? null;
  const expertName = expert?.name ?? null;
  const setContextPanelExpert = useCopilotUIStore(
    (s) => s.setContextPanelExpert,
  );

  useEffect(() => {
    setContextPanelExpert(
      expertId && expertName ? { id: expertId, name: expertName } : null,
    );
    return () => setContextPanelExpert(null);
  }, [expertId, expertName, setContextPanelExpert]);
}
