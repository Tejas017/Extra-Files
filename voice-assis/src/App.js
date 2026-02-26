// src/App.js
import React, { useState } from 'react';
import Navbar from '../../voice-as/src/components/Navbar';
import VoiceAssistant from '../../voice-as/src/components/VoiceAssistant';

function App() {
  const [listening, setListening] = useState(false);

  const toggleAssistant = () => {
    setListening(prev => !prev);
  };

  return (
    <div>
      <Navbar toggleAssistant={toggleAssistant} listening={listening} />
      <VoiceAssistant listening={listening} />
    </div>
  );
}

export default App;
