import { useSelector } from "react-redux";
import { RootState } from "../../redux/rootReducer";
import IAi from "@/types/data/IAi";

const useAiHistory: () => {
    aiHistory: Record<string, IAi>;
    aiHistoryArray: Array<IAi>;
} = () => {
    const aiHistory = useSelector((state: RootState) => state.data.aiHistory);

    return { aiHistory, aiHistoryArray: Object.values(aiHistory) }
};

export default useAiHistory;