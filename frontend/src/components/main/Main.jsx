import React from "react";
import './Main.css';

const Main = () => {
    return (
        <>
            <div className="heading-text">
                <h1>Upload an Image to check whether you have Melanoma or not for <span className="free">free! <svg className="line" width="81" height="22" viewBox="0 0 81 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M1.54187 19.9319C1.6408 18.9426 3.36948 18.3704 4.07127 17.9988C7.30292 16.2879 10.549 14.6198 13.7365 12.8269C17.5755 10.6675 21.6754 9.08432 25.7871 7.53164C33.2278 4.72186 41.1902 2.90622 49.1069 2.12325C54.8677 1.5535 60.8649 1.9793 66.6482 1.9793C69.5815 1.9793 72.5398 1.86488 75.4702 1.98958C76.0279 2.01331 78.8067 1.9688 79.0895 2.53453" stroke="#81E8FF" stroke-width="3" stroke-linecap="round"/>
                </svg>
                </span></h1>
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