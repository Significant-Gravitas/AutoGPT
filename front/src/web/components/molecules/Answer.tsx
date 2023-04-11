import IAnswer from "@/types/data/IAnswer"
import React from "react"
import styled from "styled-components"

export const AnswerContainer = styled.div`
  color: var(--grey100);
  padding: 1rem 2rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: var(--grey700);
  width: fit-content;
`

const Answer = ({ answer }: { answer: IAnswer }) => {
  return <AnswerContainer>{answer.content ?? answer.title}</AnswerContainer>
}

export default Answer
