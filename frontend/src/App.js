import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Canvas from './Canvas';
import MultiCanvas from './MultiCanvas';

export default function App() {
  return (
    <Router>
      <div className="container">
        <nav>
          <Link to="/">Single Digit</Link>
          <Link to="/multi">Multi Digit</Link>
        </nav>

        <Routes>
          <Route path="/" element={
            <>
              <header>
                <h1>Digit Recognizer</h1>
                <p className="lead">
                  Experience real-time AI. Draw a digit below and watch our neural network interpret your keystrokes instantly.
                </p>
              </header>
              <Canvas />
            </>
          } />
          <Route path="/multi" element={
            <>
              <header>
                <h1>Sequence Reader</h1>
                <p className="lead">
                  Pushing boundaries. Write a sequence of digits to test the model's ability to segment and recognize complex patterns.
                </p>
              </header>
              <MultiCanvas />
            </>
          } />
        </Routes>
      </div>
    </Router>
  );
}
