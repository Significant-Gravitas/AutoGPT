import styled from "styled-components"

export const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 3fr 1fr;
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
export const Button = styled.button`
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
  margin-left: 0.5rem;
  margin-right: 0.5rem;
  padding: 0.5rem;
  border-radius: 0.5rem;
  background-color: hsl(53, 100%, 66%);
  color: #333;
  &:hover {
    background-color: hsl(53, 100%, 46%);
    cursor: pointer;
  }
`
