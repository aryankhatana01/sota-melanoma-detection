import './App.css';
import Header from './components/header/Header';
import Main from './components/main/Main';
import Footer from './components/footer/Footer';
import PredictButton from './components/predictButton/PredictButton';

function App() {
  return (
    <>
      <div className="App">
        <Header />
        <Main />
        <PredictButton />
      </div>
      <Footer />
    </>
  );
}

export default App;
