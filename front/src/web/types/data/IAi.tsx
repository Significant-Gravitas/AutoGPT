import IAgent from "./IAgent"
import IAnswer from "./IAnswer"

interface IAi {
	id: string
	name: string
	role: string
	goals: Array<string>
	continuous: boolean
	createdAt: string
	updatedAt: string
	agents: Array<IAgent["name"]> // agent use name as id
	answers: Array<IAnswer>
}

export default IAi
