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

                <div className="upload-button">
                    <button className="upload-btn">
                        <div className="upload-svg">
                            <svg width="43" height="43" viewBox="0 0 43 43" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <g clip-path="url(#clip0_1_15)">
                                <path d="M34.6687 17.9883C34.0669 14.9384 32.4249 12.192 30.0231 10.2183C27.6213 8.24459 24.6087 7.16596 21.5 7.16667C16.3221 7.16667 11.825 10.105 9.58542 14.405C6.95209 14.6896 4.5168 15.9372 2.7475 17.9083C0.978198 19.8793 -0.000307632 22.4347 7.25495e-08 25.0833C7.25495e-08 31.0138 4.81958 35.8333 10.75 35.8333H34.0417C38.9867 35.8333 43 31.82 43 26.875C43 22.145 39.3271 18.3108 34.6687 17.9883ZM34.0417 32.25H10.75C6.79042 32.25 3.58333 29.0429 3.58333 25.0833C3.58333 21.4104 6.32458 18.3467 9.96167 17.9704L11.8787 17.7733L12.7746 16.0713C13.5985 14.4673 14.8491 13.1218 16.3886 12.183C17.9281 11.2441 19.6968 10.7482 21.5 10.75C26.1942 10.75 30.2433 14.0825 31.1571 18.6871L31.6946 21.3746L34.4358 21.5717C35.7827 21.6623 37.0453 22.2596 37.9694 23.2436C38.8935 24.2276 39.4106 25.5251 39.4167 26.875C39.4167 29.8313 36.9979 32.25 34.0417 32.25ZM14.3333 23.2917H18.9021V28.6667H24.0979V23.2917H28.6667L21.5 16.125L14.3333 23.2917Z" fill="#F8F8F8"/>
                                </g>
                                <defs>
                                <clipPath id="clip0_1_15">
                                <rect width="43" height="43" fill="white"/>
                                </clipPath>
                                </defs>
                            </svg>
                        </div>
                        <div className="upload-text">
                            Upload
                        </div>
                    </button>
                    <br />
                    <p>Maximum File size is 5MB.</p>
                </div>

                <svg className="circle2" width="314" height="314" viewBox="0 0 314 314" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="157" cy="157" r="157" fill="#606060" fill-opacity="0.1"/>
                </svg>
            </div>
        </>
    )
}

export default Main;