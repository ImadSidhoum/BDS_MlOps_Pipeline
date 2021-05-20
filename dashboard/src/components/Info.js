import React, { useEffect, useState } from "react"
import History from "./Context"

const Info = () => {
    const contextHist = React.useContext(History)
    const [scoreMean, setScoreMean] = useState(0)
    const [limit, setLimit] = useState(0)
    const [nbIn, setNbIn] = useState(0)
    const [nbOut, setNbOut] = useState(0)
    const [mIn, setMIn] = useState(0)
    const [mOut, setMOut] = useState(0)

    const requestOptions = {
        mode: "cors",
        credentials: "include",
        method: "GET",
    }

    fetch("http://localhost:5000/config", requestOptions)
        .then(response => response.json())
        .then(data => setLimit(data))

    useEffect(() => {
        let total = 0
        let in_dist = 0
        let out_dist = 0
        let avg_in = 0
        let avg_out = 0
        for (var i = 0; i < contextHist.hist.length; i++) {
            total += contextHist.hist[i].score
            if (contextHist.hist[i].label === "In Distribution") {
                in_dist += 1
                avg_in += contextHist.hist[i].score
            } else {
                out_dist += 1
                avg_out += contextHist.hist[i].score
            }
        }
        var avg = total / contextHist.hist.length
        setScoreMean(avg.toFixed(2))
        setNbIn(in_dist)
        setNbOut(out_dist)
        setMIn((avg_in / contextHist.hist.length).toFixed(2))
        setMOut((avg_out / contextHist.hist.length).toFixed(2))
    }, [contextHist.hist])

    return (
        <div className="info">
            Info
            <div className="sub-div">
                General
                <li>Size: {contextHist.hist.length}</li>
                <li>
                    Moyenne: <span>{scoreMean}</span>
                </li>
                <li>Limite: {limit}</li>
            </div>
            <div className="sub-div">
                In Distribution
                <li>Size: {nbIn}</li>
                <li>
                    Moyenne: <span>{mIn}</span>
                </li>
            </div>
            <div className="sub-div">
                Out of Distribution
                <li>Size: {nbOut}</li>
                <li>
                    Moyenne: <span>{mOut}</span>
                </li>
            </div>
        </div>
    )
}

export default Info
