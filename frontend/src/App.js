import React, { useState } from 'react';
import './App.css';
import Header from './components/header/Header';
import Main from './components/main/Main';
import Footer from './components/footer/Footer';
import PredictButton from './components/predictButton/PredictButton';
import FileContext from './components/contexts/FileContext';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  return (
    <>
      <div className="App">
        <Header />
        <FileContext.Provider value={selectedFile}>
          <Main setSelectedFile={setSelectedFile}/>
          <PredictButton />
        </FileContext.Provider>
      </div>
      <Footer />
    </>
  );
}

export default App;
