import logo from './logo.svg';
import './App.css';
import Home from './components/Home';
import Analyzer from './components/Analyzer'
import Footer from './components/Footer';
import Nav from './components/Nav';
import PostRequest from './components/Test'

import './static/css/analyzer.css';
import './static/css/contacts.css';
import './static/css/home.css';
import './static/css/footer.css';

function App() {
  return (
    <div className="App">
      <>
      <Nav />
      {/* <Home /> */}
      <Analyzer />
      {/* <Footer /> */}
      </>
    </div>
  );
}

export default App;
