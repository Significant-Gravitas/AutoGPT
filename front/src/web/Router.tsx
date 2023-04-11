import React from "react"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import MainPage from "@/pages/MainPage/MainPage"
import InitPage from "@/pages/InitPage/InitPage"

const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/main/:id" element={<MainPage />} />
        <Route path="*" element={<InitPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default Router
