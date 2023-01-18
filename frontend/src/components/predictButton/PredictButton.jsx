import React, { useContext, useState } from "react";
import './PredictButton.css';
import FileContext from "../contexts/FileContext";


const PredictButton = () => {
    const selectedFile = useContext(FileContext);
    const [pred, setPred] = useState(null);

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append("file", selectedFile, selectedFile.name);
        console.log(formData);
        console.log(selectedFile.name);

        const requestOptions = {
            method: 'POST',
            body: formData,
        };
        await fetch('http://localhost:8000/uploadfile', requestOptions);
        // const data = await resp.json();
    }
    const handlePredict = async () => {
        await handleUpload();
        const requestOptions = {
            method: 'GET',
        };
        const resp = await fetch(`http://localhost:8000/predict/?filename=${selectedFile.name}`, requestOptions);
        const data = await resp.json();
        setPred(data["Prediction"]);
        console.log(data);
    }
    return (
        <div className="predict-button">
            <button className="predict-button-button" onClick={handlePredict}>Predict!</button>
            {/* {Object.entries(pred).map(([key, value]) => (
                <div key={key}>{key}: {value}</div>
            ))} */}
            <div className="pred">
                {/* {pred} */}
                {pred===null ? null : (pred === 6 ? <div className="detected">Prediction: Melanoma Detected</div> : <div className="no-detected">Prediction: No Melanoma Detected</div>)}
            </div>
            {/* <div className="">
                {selectedFile}
            </div> */}
        </div>
    )
}

export default PredictButton;