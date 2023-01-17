import React, { useContext } from "react";
import './PredictButton.css';
import FileContext from "../contexts/FileContext";


const PredictButton = () => {
    const selectedFile = useContext(FileContext);
    const handleUpload = () => {
        const formData = new FormData();
        formData.append("file", selectedFile, selectedFile.name);
        console.log(formData);
        console.log(selectedFile.name);

        const requestOptions = {
            method: 'POST',
            body: formData,
        };
        fetch('http://localhost:8000/uploadfile', requestOptions)
    }
    return (
        <div className="predict-button">
            <button className="predict-button-button" onClick={handleUpload}>Predict!</button>
            {/* <div className="">
                {selectedFile}
            </div> */}
        </div>
    )
}

export default PredictButton;