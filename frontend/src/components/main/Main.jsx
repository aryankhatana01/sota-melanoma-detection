import React from "react";
import './Main.css';

const Main = () => {
    return (
        <>
            <div className="heading-text">
                <h1>Upload an Image to check whether you have Melanoma or not for <span className="free">free!</span></h1>
            </div>
            <div className="upload-section">
                <svg className="circle1" width="314" height="314" viewBox="0 0 314 314" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="157" cy="157" r="157" fill="#606060" fill-opacity="0.1"/>
                </svg>

                <svg className="circle2" width="314" height="314" viewBox="0 0 314 314" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="157" cy="157" r="157" fill="#606060" fill-opacity="0.1"/>
                </svg>
            </div>
        </>
    )
}

export default Main;