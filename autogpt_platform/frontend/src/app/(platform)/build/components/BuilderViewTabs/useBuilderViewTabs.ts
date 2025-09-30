import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo } from "react";
import { BuilderView } from "./BuilderViewTabs";

export function useBuilderView() {
  const isNewFlowEditorEnabled = useGetFlag(Flag.NEW_FLOW_EDITOR);
  const isBuilderViewSwitchEnabled = useGetFlag(Flag.BUILDER_VIEW_SWITCH);

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const currentView = searchParams.get("view");
  const defaultView = "old";
  const selectedView = useMemo<BuilderView>(() => {
    if (currentView === "new" || currentView === "old") return currentView;
    return defaultView;
  }, [currentView, defaultView]);

  useEffect(() => {
    if (isBuilderViewSwitchEnabled === true) {
      if (currentView !== "new" && currentView !== "old") {
        const params = new URLSearchParams(searchParams);
        params.set("view", defaultView);
        router.replace(`${pathname}?${params.toString()}`);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBuilderViewSwitchEnabled, defaultView, pathname, router, searchParams]);

  const setSelectedView = (value: BuilderView) => {
    const params = new URLSearchParams(searchParams);
    params.set("view", value);
    router.push(`${pathname}?${params.toString()}`);
  };

  return {
    isSwitchEnabled: isBuilderViewSwitchEnabled === true,
    selectedView,
    setSelectedView,
    isNewFlowEditorEnabled: Boolean(isNewFlowEditorEnabled),
  } as const;
}
