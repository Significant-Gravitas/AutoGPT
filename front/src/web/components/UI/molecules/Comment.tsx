import IAnswer from "../../../types/data/IAnswer";
import React from "react";
import styled from "styled-components";

export const CommentContainer = styled.div`
  color: #454545;
  font-size: 0.8rem;
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
  margin-left: 0.5rem;
  margin-right: 0.5rem;
  padding: 1rem 2rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: #eee;
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-all;
  word-wrap: break-word;
  width: fit-content;
`;

const Comment = ({ comment }: { comment: IAnswer }) => {
	return (
		<CommentContainer>{comment.content ?? comment.title}</CommentContainer>
	);
};

export default Comment;
