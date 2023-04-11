import { RootState } from '@/redux/rootReducer';
import IAgent from '@/types/data/IAgent';
import { useSelector } from 'react-redux';

const useAgents: () => {
    agents: RootState["data"]["agents"];
    agentsArray: Array<IAgent>;
} = () => {
    const agents = useSelector((state: RootState) => state.data.agents)

    return { agents, agentsArray: Object.values(agents) };
}

export default useAgents;