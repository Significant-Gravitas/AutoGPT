import IAnswer from '@/types/data/IAnswer'
import AutoGPTAPI from '../api/AutoGPTAPI'
import { useDispatch } from 'react-redux'
import { addAnswersToAi } from '@/redux/data/dataReducer'
const useAutoGPTAPI = () => {
    const dispatch = useDispatch()
    const fetchData = async () => {
        const data = await AutoGPTAPI.fetchData() as Array<IAnswer>
        dispatch(addAnswersToAi
    }
}