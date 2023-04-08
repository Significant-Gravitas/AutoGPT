import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import MainPage from "./components/pages/MainPage/MainPage";
import LeftPanel from "./components/UI/organisms/LeftPanel";

const Router = () => {
	return (
		<BrowserRouter>
			<Routes>
				<Route path="*" element={<MainPage />} />
			</Routes>
		</BrowserRouter>
	);
};

export default Router;
