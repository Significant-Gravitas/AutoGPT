import useAiHistory from "@/hooks/data/useAiHistory"
import Flex from "@/style/Flex"
import { useNavigate, useParams } from "react-router"
import styled from "styled-components"
import SearchInput from "../../molecules/SearchInput"
import TaskCard from "../../molecules/TaskCard"
import SButton from "@/style/SButton"
import { AiCardContainer, AiListContainer } from "./AiList.styled"
import { useMemo, useState } from "react"

const AiList = () => {
	const { aiHistoryArray } = useAiHistory()
	const navigate = useNavigate()
	const { id } = useParams<{ id: string }>()
	const [search, setSearch] = useState("")
	const filteredAiHistoryArray = useMemo(() => {
		if (search === "") {
			return aiHistoryArray
		}
		return aiHistoryArray.filter((ai) => {
			return ai.name.toLowerCase().includes(search.toLowerCase())
		})
	}, [aiHistoryArray, search])

	return (
		<AiListContainer>
			<Flex direction="column" gap={1}>
				<h2>All your AI</h2>
				<SearchInput
					value={search}
					onChange={(e) => setSearch(e.target.value)}
				/>
				<AiCardContainer>
					{filteredAiHistoryArray.map((ai) => (
						<TaskCard ai={ai} key={ai.id} $active={id === ai.id} />
					))}
					{filteredAiHistoryArray.length === 0 && <div>No Ai found</div>}
				</AiCardContainer>
				<SButton
					$color="primary300"
					variant="outlined"
					onClick={() => navigate("/")}
				>
					Create a new Ai
				</SButton>
			</Flex>
		</AiListContainer>
	)
}

export default AiList
