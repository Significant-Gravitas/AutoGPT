import { useEffect, useState } from "react"
import IAnswer from "../types/data/IAnswer"
import IAgent from "../types/data/IAgent"

const AGENT_CREATED = "COMMAND = start_agent  ARGUMENTS = "

const useAnswerInterceptor = (data: IAnswer[]) => {
  const [agents, setAgents] = useState<IAgent[]>([])

  const interceptAnswer = (data: IAnswer[]) => {
    data.forEach((answer) => {
      if (!answer.content) return
      debugger
      if (answer.content.includes(AGENT_CREATED)) {
        let agentSystemReturn = answer.content.split(AGENT_CREATED)[1]
        const jsonAgent = JSON.parse(
          agentSystemReturn.replace(/'/g, '"'),
        ) as IAgent

        if (agents.find((agent) => agent.name === jsonAgent.name)) return
        setAgents([...agents, jsonAgent])
      }
    })
  }

  useEffect(() => {
    interceptAnswer(data)
  }, [data, agents])

  return { agents }
}

export default useAnswerInterceptor
