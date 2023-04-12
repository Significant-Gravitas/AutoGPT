import IAgent from "@/types/data/IAgent"
import IAi from "@/types/data/IAi"
import IAnswer from "@/types/data/IAnswer"
import {
  ActionCreatorWithPayload,
  PayloadAction,
  SliceCaseReducers,
  createSlice,
} from "@reduxjs/toolkit"

interface IDataState {
  aiHistory: Record<string, IAi>
  agents: Record<string, IAgent>
}

const dataReducer = createSlice<
  IDataState,
  SliceCaseReducers<IDataState>,
  "data"
>({
  name: "data",
  initialState: {
    aiHistory: {},
    agents: {},
  },
  reducers: {
    addAiHistory: (state: IDataState, action: PayloadAction<IAi>) => {
      state.aiHistory[action.payload.id] = action.payload
    },
    addAgent: (
      state: IDataState,
      action: PayloadAction<{ agent: IAgent; aiId: IAi["id"] }>,
    ) => {
      state.agents[action.payload.agent.name] = action.payload.agent
      state.aiHistory[action.payload.aiId].agents.push(
        action.payload.agent.name,
      )
    },
    addAnswersToAi: (
      state: IDataState,
      action: PayloadAction<{ aiId: string; answers: Array<IAnswer> }>,
    ) => {
      state.aiHistory[action.payload.aiId].answers = [
        ...state.aiHistory[action.payload.aiId].answers,
        ...action.payload.answers,
      ]
    },
    deleteAi: (state: IDataState, action: PayloadAction<IAi["id"]>) => {
      delete state.aiHistory[action.payload]
    },
    removeAgent: (state: IDataState, action: PayloadAction<IAgent["name"]>) => {
      delete state.agents[action.payload]
    },
  },
})

export const { addAiHistory, addAgent, removeAgent, addAnswersToAi, deleteAi } =
  dataReducer.actions as {
    addAiHistory: ActionCreatorWithPayload<IAi>
    addAgent: ActionCreatorWithPayload<{
      agent: IAgent
      aiId: IAi["id"]
    }>
    removeAgent: ActionCreatorWithPayload<IAgent["name"]>
    addAnswersToAi: ActionCreatorWithPayload<{
      aiId: string
      answers: Array<IAnswer>
    }>
    deleteAi: ActionCreatorWithPayload<IAi["id"]>
  }
export default dataReducer.reducer
