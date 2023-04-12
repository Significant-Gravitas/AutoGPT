import { useEffect, useState } from "react"
import IAnswer from "../types/data/IAnswer"
import IAgent from "../types/data/IAgent"
import { useDispatch } from "react-redux"
import { addAgent, removeAgent } from "@/redux/data/dataReducer"
import useAgents from "./data/useAgents"
import { useParams } from "react-router"

const AGENT_CREATED = "COMMAND = start_agent  ARGUMENTS = "
const DELETE_AGENT = "COMMAND = delete_agent  ARGUMENTS = "
const WRITE_FILE = "COMMAND = write_to_file ARGUMENTS = "
const BROWSE_WEBSITE = "COMMAND = browse_website ARGUMENTS = "
const GOOGLE_RETURN = "Command google returned: "
const useAnswerInterceptor = () => {
  const dispatch = useDispatch()
  const { id } = useParams<{ id: string }>()
  const { agents, agentsArray } = useAgents()

  const interceptAnswer = (data: IAnswer[]) => {
    if (!id) return
    data.forEach((answer) => {
      if (!answer.content) return
      if (answer.content.includes(AGENT_CREATED)) {
        let agentSystemReturn = answer.content.split(AGENT_CREATED)[1]
        const jsonAgent = JSON.parse(
          agentSystemReturn.replace(/'/g, '"'),
        ) as IAgent

        if (agentsArray.find((agent) => agent.name === jsonAgent.name)) return
        dispatch(addAgent({ agent: jsonAgent, aiId: id }))
      }
      if (answer.content.includes(DELETE_AGENT)) {
        let agentSystemReturn = answer.content.split(DELETE_AGENT)[1]
        const jsonAgent = JSON.parse(
          agentSystemReturn.replace(/'/g, '"'),
        ) as IAgent

        dispatch(removeAgent(jsonAgent.name))
      }
    })
  }

  return {
    interceptAnswer,
  }
}

export default useAnswerInterceptor
