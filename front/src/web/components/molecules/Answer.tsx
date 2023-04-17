import AutoGPTAPI from "@/api/AutoGPTAPI"
import { SIconButton } from "@/pages/MainPage/MainPage.styled"
import Flex from "@/style/Flex"
import colored from "@/style/colored"
import IAnswer, { InternalType } from "@/types/data/IAnswer"
import { FileCopy, Sync } from "@mui/icons-material"
import styled from "styled-components"
import { Card } from "../atom/Card"
import Details from "../atom/Details"
import CardContent from "../atom/CardContent"
import LineSeparatorWithTitle from "./LineSeparatorWithTitle"
import { Avatar } from "@mui/material"

const SFileCopy = colored(styled(FileCopy)`
  color: var(--color) !important;
`)

const RotatingArrow = styled(Sync)`
  animation: spin 2s linear infinite;
  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`
const ThinkingContainer = styled.div`
  margin-top: 1rem;
  margin-bottom: 1rem;
`
export const AnswerContainer = styled.div`
  color: var(--grey100);
  padding: 1rem 2rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: var(--grey700);
  width: fit-content;
`

export const SAvatar = styled(Avatar)`
  border: 1px solid var(--primary300);
  border-radius: 50%;
  width: 3rem;
  height: 3rem;
`

const Answer = ({
	answer,
	isAnswerLast,
	playing,
}: {
	answer: IAnswer
	isAnswerLast: boolean
	playing: boolean
}) => {
	switch (answer.internalType) {
		case InternalType.THINKING:
			if (isAnswerLast && playing) {
				return (
					<Flex gap={1}>
						<SAvatar
							src="/images/autoctopus.png"
							sx={{ width: 42, height: 42 }}
						/>
						<ThinkingContainer>
							<Details $color="grey100">
								<Flex align="center" gap={0.5}>
									<RotatingArrow />
									<div>Thinking</div>
								</Flex>
							</Details>
						</ThinkingContainer>
					</Flex>
				)
			}
			return null
		case InternalType.PLAN:
			return (
				<>
					<LineSeparatorWithTitle title={answer.title} />
					<Flex gap={1}>
						<SAvatar
							src="/images/autoctopus.png"
							sx={{ width: 42, height: 42 }}
						/>
						<AnswerContainer>
							<Flex direction="column" gap={0.5}>
								{answer.content.split("-").map((item, index) => (
									<div key={index}>- {item}</div>
								))}
							</Flex>
						</AnswerContainer>
					</Flex>
				</>
			)
		case InternalType.WRITE_FILE:
			return (
				<>
					<LineSeparatorWithTitle title={answer.title} />
					<Flex gap={1}>
						<SAvatar
							src="/images/autoctopus.png"
							sx={{ width: 42, height: 42 }}
						/>
						<Card
							$fitContent
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
								<Flex gap={0.5} align="center">
									<SFileCopy $color="primary" />
									<Details $color="primary">
										{JSON.parse(answer.content.replaceAll("'", '"')).file}
									</Details>
								</Flex>
							</CardContent>
						</Card>
					</Flex>
				</>
			)
		default:
			return (
				<>
					<LineSeparatorWithTitle title={answer.title} />
					<Flex gap={1}>
						<SAvatar
							src="/images/autoctopus.png"
							sx={{ width: 42, height: 42 }}
						/>
						<AnswerContainer>{answer.content ?? answer.title}</AnswerContainer>
					</Flex>
				</>
			)
	}
}

export default Answer
