// src/components/Navbar.js
import React from 'react';

const Navbar = ({ toggleAssistant, listening }) => {
  return (
    <nav style={{ padding: '1rem', backgroundColor: '#282c34', color: 'white' }}>
      <h2 style={{ display: 'inline-block', marginRight: '2rem' }}>🎙️ Voice Assistant</h2>
      <button onClick={toggleAssistant}>
        {listening ? '🛑 Stop Assistant' : '🎙️ Start Assistant'}
      </button>
    </nav>
  );
};

export default Navbar;
