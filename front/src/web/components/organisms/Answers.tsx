import Flex from "@/style/Flex"
import IAnswer from "@/types/data/IAnswer"
import styled from "styled-components"
import Answer from "../molecules/Answer"

export const AnswerContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-right: 1rem;
`

const Answers = ({
	answers,
	playing,
}: {
	answers: IAnswer[]
	playing: boolean
}) => {
	if (!answers) return null
	if (!Array.isArray(answers)) return null
	return (
		<AnswerContainer>
			{answers.map((answer, index) => (
				<Answer
					answer={answer}
					key={answer.title}
					isAnswerLast={index === answers.length - 1}
					playing={playing}
				/>
			))}
		</AnswerContainer>
	)
}

export default Answers
