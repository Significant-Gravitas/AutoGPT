import { useEffect, useState } from "react"
import IAnswer, { InternalType } from "../types/data/IAnswer"
import IAgent from "../types/data/IAgent"
import { useDispatch } from "react-redux"
import { addAgent, removeAgent } from "@/redux/data/dataReducer"
import useAgents from "./data/useAgents"
import { useParams } from "react-router"

const AGENT_CREATED = "COMMAND = start_agent  ARGUMENTS = "
const DELETE_AGENT = "COMMAND = delete_agent  ARGUMENTS = "
const WRITE_FILE = "COMMAND = write_to_file  ARGUMENTS = "
const BROWSE_WEBSITE = "COMMAND = browse_website  ARGUMENTS = "
const GOOGLE_RETURN = "Command google returned: "
const APPEND_FILE = "COMMAND = append_to_file  ARGUMENTS = "
const useAnswerInterceptor = () => {
	const dispatch = useDispatch()
	const { id } = useParams<{ id: string }>()
	const { agents, agentsArray } = useAgents()

	const interceptAnswer = (data: IAnswer[]) => {
		if (!id) return data
		const newData = [] as Array<IAnswer>
		debugger
		data.forEach((answer) => {
			if (!answer.content) return newData.push(answer)

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
				debugger
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
