import {
  createContext,
  use,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";

import { useSignedIn } from "../auth/SessionProvider";
import { forgetToolParts } from "../chat/tools/detailsStore";
import { ChatListStore } from "../chats/chatListStore";
import { ExpertDirectory } from "../company/expertDirectory";

interface AppData {
  chats: ChatListStore;
  experts: ExpertDirectory;
}

const AppDataContext = createContext<AppData | null>(null);

/**
 * Caches that belong to one signed-in account on one server. Mount it keyed
 * by account so nothing can carry over to another account or server.
 */
export function AppDataProvider({ children }: { children: ReactNode }) {
  const { api } = useSignedIn();
  const [stores] = useState<AppData>(() => ({
    chats: new ChatListStore(api),
    experts: new ExpertDirectory(api),
  }));

  useEffect(() => {
    stores.experts.ensureLoaded();
    return () => forgetToolParts();
  }, [stores]);

  return <AppDataContext value={stores}>{children}</AppDataContext>;
}

function useAppData(): AppData {
  const value = use(AppDataContext);
  if (!value) throw new Error("useAppData must be used inside AppDataProvider");
  return value;
}

export function useChatList() {
  const { chats } = useAppData();
  const state = useSyncExternalStore(chats.subscribe, chats.getSnapshot);
  return { state, store: chats };
}

export function useExperts() {
  const { experts } = useAppData();
  const state = useSyncExternalStore(experts.subscribe, experts.getSnapshot);
  return { state, directory: experts };
}
