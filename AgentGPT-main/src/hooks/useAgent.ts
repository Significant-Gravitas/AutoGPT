import { api } from "../utils/api";
import type { Message } from "../types/agentTypes";
import { useAuth } from "./useAuth";

export interface SaveProps {
  name: string;
  goal: string;
  tasks: Message[];
}

export function useAgent() {
  const { status } = useAuth();
  const utils = api.useContext();
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  const voidFunc = () => {};
  const saveMutation = api.agent.create.useMutation({
    onSuccess: (data) => {
      utils.agent.getAll.setData(voidFunc(), (oldData) => [
        data,
        ...(oldData ?? []),
      ]);
    },
  });

  const saveAgent = (data: SaveProps) => {
    if (status === "authenticated") saveMutation.mutate(data);
  };

  return {
    saveAgent,
  };
}
