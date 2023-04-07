import styled from "styled-components"

export const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 3fr 1fr;
  grid-gap: 1rem;
`
export const Comment = styled.div`
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
`
export const Input = styled.input`
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
`
export const Container = styled.div`
  background-color: var(--grey800);
  color: var(--grey100);
  height: 100vh;
`
export const RightTasks = styled.div`
  background-color: var(--grey800);
  color: var(--grey100);
  height: 100%;
  border-radius: 0.5rem;
  padding: 1rem;
`
export const CardTask = styled.div`
  color: var(--grey100);
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  &:hover {
    background-color: var(--grey700);
    cursor: pointer;
  }
  & > p {
    margin: 0;
    font-size: 0.8rem;
    color: var(--grey300);
  }
`
export const Search = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  width: 100%;
  & > input {
    padding: 0.5rem;
    border-radius: 0.5rem;
    background-color: var(--grey700);
    color: var(--grey100);
    border: none;
    padding: 0.75rem;
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-all;
    word-wrap: break-word;
    width: 100%;
    font-size: 1rem;
  }
`
export const Line = styled.div<{ content?: string }>`
  width: 100%;
  height: 1px;
  background-color: var(--grey100);
  margin-bottom: 1rem;
  margin-top: 1rem;
  // text in middle
  &::before {
    content: "${({ content }) => content}";
    position: relative;
    top: -0.5rem;
    right: -50%;
    display: inline-block;
    padding: 0 0.5rem;
    background-color: var(--grey700);
    transform: translateX(-50%);
  }
`
export const Discussion = styled.div`
  padding: 2rem;
  background-color: var(--grey700);
`
