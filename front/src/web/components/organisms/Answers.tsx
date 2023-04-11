import IAnswer from "@/types/data/IAnswer"
import styled from "styled-components"
import Answer from "../molecules/Answer"
import LineSeparatorWithTitle from "../molecules/LineSeparatorWithTitle"

export const AnswerContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`

const Answers = ({ answers }: { answers: IAnswer[] }) => {
  if (!answers) return null
  if (!Array.isArray(answers)) return null
  return (
    <AnswerContainer>
      {answers.map((answer) => (
        <>
          <LineSeparatorWithTitle title={answer.title} />
          <Answer answer={answer} />
        </>
      ))}
    </AnswerContainer>
  )
}

export default Answers
