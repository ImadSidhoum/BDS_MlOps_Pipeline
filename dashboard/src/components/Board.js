import React, { useState } from "react"
import Allocation from "./Allocation"
import Performance from "./Performance"
import Info from "./Info"
import Position from "./Position"
import History from "./Context"

const Board = () => {
    const [hist, setHist] = useState([])

    const contextHist = {
        hist: hist,
        setHist: setHist,
    }

    return (
        <div className="board">
            <History.Provider value={contextHist}>
                <Info />
                <Allocation />
                <Performance />
                <Position />
            </History.Provider>
        </div>
    )
}

export default Board
