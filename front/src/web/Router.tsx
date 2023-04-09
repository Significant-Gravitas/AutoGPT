import React from "react"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import MainPage from "./components/pages/MainPage/MainPage"
import LeftPanel from "./components/UI/organisms/LeftPanel"
import InitPage from "./components/pages/InitPage/InitPage"

const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/main" element={<MainPage />} />
        <Route path="*" element={<InitPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default Router
