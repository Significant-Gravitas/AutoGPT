import AutoGPTAPI from "@/api/AutoGPTAPI"
import { SIconButton } from "@/pages/MainPage/MainPage.styled"
import Flex from "@/style/Flex"
import colored from "@/style/colored"
import IAnswer, { InternalType } from "@/types/data/IAnswer"
import { FileCopy } from "@mui/icons-material"
import styled from "styled-components"
import { Card } from "../atom/Card"
import Details from "../atom/Details"
import CardContent from "../atom/CardContent"

const SFileCopy = colored(styled(FileCopy)`
  color: var(--color) !important;
`)

export const AnswerContainer = styled.div`
  color: var(--grey100);
  padding: 1rem 2rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: var(--grey700);
  width: fit-content;
`

const Answer = ({ answer }: { answer: IAnswer }) => {
	switch (answer.internalType) {
		case InternalType.WRITE_FILE:
			return (
				<AnswerContainer>
					<h3>Write to file :</h3>
					<Card
						$borderColor="primary"
						$textColor="primary300"
						$color="grey700"
						$noPadding
						onClick={() => {
							const content = JSON.parse(answer.content.replaceAll("'", '"'))
							AutoGPTAPI.downloadFile(content.file)
						}}
					>
						<CardContent>
							<Flex gap={2} align="center">
								<SFileCopy $color="primary" />
								<Details $color="primary">
									{JSON.parse(answer.content.replaceAll("'", '"')).file}
								</Details>
							</Flex>
						</CardContent>
					</Card>
				</AnswerContainer>
			)
		default:
			return <AnswerContainer>{answer.content ?? answer.title}</AnswerContainer>
	}
}

export default Answer
