import React from "react"
import styled from "styled-components"
import IAnswer from "../../../types/data/IAnswer"
import LineSeparatorWithTitle from "../molecules/LineSeparatorWithTitle"
import Comment from "../molecules/Comment"

export const CommentContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`

const Comments = ({ comments }: { comments: IAnswer[] }) => {
  if (!comments) return null
  if (!Array.isArray(comments)) return null
  return (
    <CommentContainer>
      {comments.map((comment) => (
        <>
          <LineSeparatorWithTitle title={comment.title} />
          <Comment comment={comment} />
        </>
      ))}
    </CommentContainer>
  )
}

export default Comments
