import { Paper, TextField } from "@mui/material";
import styled from "styled-components";

export const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 3fr 1fr;
  grid-gap: 1rem;
`;

export const Input = styled(TextField)`
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
  margin-left: 0.5rem;
  margin-right: 0.5rem;
  padding: 0.5rem;
  border: 1px solid #999;
  border-radius: 0.5rem;
  background-color: #eee;
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-all;
  word-wrap: break-word;
  min-height: 5rem;
  width: 100%;
`;
export const Container = styled.div`
  background-color: var(--grey800);
  color: var(--grey100);
  height: 100vh;
  overflow: hidden;
`;
export const RightTasks = styled.div`
  background-color: var(--grey800);
  color: var(--grey100);
  height: 100%;
  border-radius: 0.5rem;
  padding: 1rem;
`;
export const CardTask = styled(Paper)`
  color: var(--grey100) !important;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  background-color: var(--grey800) !important;
  &:hover {
    background-color: var(--grey700) !important;
    cursor: pointer;
  }
  & > p {
    margin: 0;
    font-size: 0.8rem;
    color: var(--grey300);
  }
`;
export const Search = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  width: 100%;
  & input {
    padding: 0.5rem;
    border-radius: 0.5rem;
    background-color: var(--grey700) !important;
    color: var(--grey100) !important;
    border-color: var(--grey100) !important;
    border: none;
    padding: 0.75rem;
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-all;
    word-wrap: break-word;
    width: 100%;
    font-size: 1rem;
  }
`;

export const Discussion = styled.div`
  --action-bar-height: 4rem;
  --input-container-height: 8rem;

  overflow-y: auto;
  background-color: var(--grey700);
  position: relative;
`;
export const ActionBar = styled.div`
  position: sticky;
  background-color: var(--grey500);
  padding: 1rem;
  top: 0;
  width: 100%;
  display: flex;
  gap: 1rem;
  height: var(--action-bar-height);
  z-index: 2;
`;
export const InputContainer = styled.div`
  position: sticky;
  bottom: 0;
  width: 100%;
  display: flex;
  gap: 1rem;
  height: var(--input-container-height);
  z-index: 2;
  background-color: var(--grey500);
  padding: 1rem;
  width: 100%;
`;
export const CommentContainer = styled.div`
  max-height: calc(
    100vh - var(--action-bar-height) - var(--input-container-height)
  );
  overflow-y: auto;
`;
