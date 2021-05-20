import React, { useCallback, useState } from "react"
import car from "../assets/car.png"
import History from "./Context"

const Allocation = () => {
    const [img, setImg] = useState({ file: null, img: null, label: "None", score: "inf" })
    const contextHist = React.useContext(History)

    const wrap_set_img = useCallback(
        elt => {
            setImg(elt)
            if (elt.score !== "inf") {
                contextHist.setHist([...contextHist.hist, elt])
            }
        },
        [contextHist],
    )

    const load_img = event => {
        console.log(event)
        wrap_set_img({
            file: event.target.files[0],
            img: URL.createObjectURL(event.target.files[0]),
            label: "None",
            score: "inf",
        })
    }

    const submitChange = () => {
        const data = new FormData()
        data.append("file", img.file)

        const requestOptions = {
            mode: "cors",
            credentials: "include",
            method: "POST",
            body: data,
        }

        fetch("http://localhost:5000/predict", requestOptions)
            .then(response => response.json())
            .then(data =>
                wrap_set_img({
                    file: img.file,
                    img: img.img,
                    label:
                        data["label"] === 1 ? "In Distribution" : "Out of Distribution",
                    score: Number(data["loss"]),
                }),
            )
    }

    return (
        <div className="allocation">
            Validation de donnée
            <div className="allo_grid">
                <div className="img_loader">
                    <img src={img.img == null ? car : img.img} alt="img-aveugle" />
                    <input
                        className="input-file"
                        type="file"
                        onChange={e => load_img(e)}
                    />
                    <button className="button" type="file" onClick={() => submitChange()}>
                        <span>Submit </span>
                    </button>
                </div>

                <div className="stats">
                    <div className="content">
                        <li>Nom: {img.file == null ? "" : img.file.name} </li>
                    </div>
                    <div className="content">
                        <li>Label: {img.label} </li>
                    </div>
                    <div className="content">
                        <li>Score: {img.score} </li>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Allocation
