import { useEffect, useState } from "react"
import IAnswer, { InternalType } from "../types/data/IAnswer"
import IAgent from "../types/data/IAgent"
import { useDispatch } from "react-redux"
import { addAgent, removeAgent } from "@/redux/data/dataReducer"
import useAgents from "./data/useAgents"
import { useParams } from "react-router"

const AGENT_CREATED = "COMMAND = start_agent ARGUMENTS = "
const DELETE_AGENT = "COMMAND = delete_agent ARGUMENTS = "
const WRITE_FILE = "COMMAND = write_to_file ARGUMENTS = "
const PLAN = "PLAN"
const THINKING = "Thinking"
const BROWSE_WEBSITE = "COMMAND = browse_website ARGUMENTS = "
const GOOGLE_RETURN = "Command google returned: "
const APPEND_FILE = "COMMAND = append_to_file ARGUMENTS = "
const useAnswerInterceptor = () => {
  const dispatch = useDispatch()
  const { id } = useParams<{ id: string }>()
  const { agents, agentsArray } = useAgents()

  const interceptAnswer = (data: IAnswer[]) => {
    if (!id) return data
    const newData = [] as Array<IAnswer>
    data.forEach((answer) => {
      if (answer.title.includes(THINKING)) {
        newData.push({
          title: "Thinking",
          content: answer.content,
          internalType: InternalType.THINKING,
        })
        return
      }
      if (answer.title.includes(PLAN)) {
        newData.push({
          title: "Plan",
          content: answer.content,
          internalType: InternalType.PLAN,
        })
        return
      }

      if (answer.content.includes(AGENT_CREATED)) {
        const agentSystemReturn = answer.content.split(AGENT_CREATED)[1]
        const jsonAgent = JSON.parse(
          agentSystemReturn.replace(/'/g, '"'),
        ) as IAgent

        if (agentsArray.find((agent) => agent.name === jsonAgent.name))
          return newData.push(answer)
        dispatch(addAgent({ agent: jsonAgent, aiId: id }))
        newData.push(answer)
        return
      }

      if (answer.content.includes(DELETE_AGENT)) {
        const agentSystemReturn = answer.content.split(DELETE_AGENT)[1]
        const jsonAgent = JSON.parse(
          agentSystemReturn.replace(/'/g, '"'),
        ) as IAgent

        dispatch(removeAgent(jsonAgent.name))
        newData.push(answer)
        return
      }

      if (answer.content.includes(WRITE_FILE)) {
        newData.push({
          title: "Write to file",
          content: answer.content.split(WRITE_FILE)[1],
          internalType: InternalType.WRITE_FILE,
        })
        return
      }

      newData.push(answer)
    })
    return newData
  }

  return {
    interceptAnswer,
  }
}

export default useAnswerInterceptor
