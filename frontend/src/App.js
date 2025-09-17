import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Canvas from './Canvas';
import MultiCanvas from './MultiCanvas';

export default function App() {
  return (
    <Router>
      <div className="container">
        <h1>Digit Recognizer</h1>
        <nav>
          <Link to="/">Single Digit</Link> |{' '}
          <Link to="/multi">Multi Digit</Link>
        </nav>
        <Routes>
          <Route path="/" element={<Canvas />} />
          <Route path="/multi" element={<MultiCanvas />} />
        </Routes>
      </div>
    </Router>
  );
}
