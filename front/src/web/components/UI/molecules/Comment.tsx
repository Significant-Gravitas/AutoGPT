import IAnswer from "../../../types/data/IAnswer"
import React from "react"
import styled from "styled-components"

export const CommentContainer = styled.div`
  color: var(--grey100);
  padding: 1rem 2rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: var(--grey700);
  width: fit-content;
`

const Comment = ({ comment }: { comment: IAnswer }) => {
  return <CommentContainer>{comment.content ?? comment.title}</CommentContainer>
}

export default Comment
