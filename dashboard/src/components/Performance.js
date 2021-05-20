import React from "react"
// import {useEffect, useState} from 'react';
import History from "./Context"
import { Timeline } from "antd"

const Performance = () => {
    const contextHist = React.useContext(History)

    return (
        <div className="performance">
            Images
            <div className="timeline">
                <Timeline>
                    {contextHist.hist
                        .filter((_, i) => contextHist.hist.length - i < 7)
                        .map(e => {
                            return (
                                <Timeline.Item
                                    color={
                                        e.label === "In Distribution" ? "green" : "red"
                                    }
                                >
                                    {e.file.name}{" "}
                                </Timeline.Item>
                            )
                        })}
                </Timeline>
            </div>
        </div>
    )
}

export default Performance
