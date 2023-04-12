import IAnswer from "@/types/data/IAnswer"
import AutoGPTAPI from "../api/AutoGPTAPI"
import { useDispatch } from "react-redux"
import { addAnswersToAi } from "@/redux/data/dataReducer"
import { useParams } from "react-router"
import useAnswerInterceptor from "./useAnswerInterceptor"

const useAutoGPTAPI = () => {
	const dispatch = useDispatch()
	const { id } = useParams<{ id: string }>()
	const { interceptAnswer } = useAnswerInterceptor()

	const fetchData = async () => {
		if (!id) return
		let data = (await AutoGPTAPI.fetchData()) as Array<IAnswer>
		if (data.length === 0) return
		data = interceptAnswer(data)
		dispatch(
			addAnswersToAi({
				aiId: id,
				answers: data,
			}),
		)
	}

	return {
		fetchData,
	}
}

export default useAutoGPTAPI
